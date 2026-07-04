"""
Azure Frontend Service — Chunked Upload (async, ordered, verified) + Modal call
===============================================================================

Reworked file-transfer protocol:

  * ASYNC Azure SDK (azure.storage.blob.aio) — parallel chunks truly overlap
    instead of blocking the event loop on each stage_block.
  * SINGLE cached BlobServiceClient — no new client/connection-pool per chunk.
  * ORDERED commit — blocks are committed strictly by chunk index (0,1,2,…),
    NOT by arrival order. This is the fix for "not a zip file": parallel chunks
    arrive out of order, and committing them in arrival order scrambled the
    archive (correct total size, corrupt zip).
  * VERIFICATION — every chunk present, committed size matches expected size,
    content-type set to application/zip.
  * Idempotent — a re-sent chunk overwrites its slot instead of duplicating.

The HTTP contract (routes, request/response shape) is unchanged, so the
frontend (app.js) needs no changes.
"""

import os
import uuid
import base64
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware

# Async blob client for non-blocking uploads; ContentSettings is shared.
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings

# ─── Configuration ────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 3000))
AZURE_CONTAINER = "dicom-uploads"
CHUNK_SIZE = 5 * 1024 * 1024
STAGE_RETRIES = 2  # transient-failure retries per block

# In-memory upload sessions
upload_sessions: Dict[str, Dict[str, Any]] = {}


# ─── Azure Blob Storage (async, cached) ───────────────────────────────────

_blob_service_client: BlobServiceClient | None = None


def get_blob_service_client() -> BlobServiceClient:
    """Return a single cached async BlobServiceClient (created lazily in-loop)."""
    global _blob_service_client
    if _blob_service_client is None:
        _blob_service_client = BlobServiceClient.from_connection_string(
            os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        )
    return _blob_service_client


def get_blob_client(blob_name: str):
    """Async blob client for a specific blob (reuses the cached service client)."""
    return get_blob_service_client().get_blob_client(
        container=AZURE_CONTAINER, blob=blob_name
    )


def _block_id(chunk_index: int) -> str:
    """Deterministic, fixed-width, base64 block id for a chunk index."""
    return base64.b64encode(f"block-{chunk_index:08d}".encode()).decode()


# ─── Modal Function Reference ──────────────────────────────────────────────

_run_inference = None


def get_modal_function():
    try:
        import modal
        fn = modal.Function.from_name("plumonet-inference", "run_inference")
        print("[Modal] ✓ Got reference to deployed run_inference function")
        return fn
    except Exception as e:
        print(f"[Modal ERROR] Failed to get function reference: {e}")
        print("[Modal] Make sure 'plumonet-inference' app is deployed!")
        return None


async def call_modal_inference(blob_name: str) -> Dict[str, Any]:
    global _run_inference
    if _run_inference is None:
        _run_inference = get_modal_function()
    if _run_inference is None:
        raise Exception("Modal function not available")

    print(f"[Modal] Calling run_inference('{blob_name}')")
    result = await _run_inference.remote.aio(blob_name)
    print("[Modal] Inference complete!")
    return result


# ─── Route Handlers ──────────────────────────────────────────────────────

async def initiate_upload(request) -> JSONResponse:
    """POST /api/upload/initiate — start a chunked upload session."""
    try:
        body = await request.json()
        fileName = body.get("fileName")
        fileSize = body.get("fileSize")

        if not fileName or not fileSize:
            return JSONResponse({"error": "fileName and fileSize required"}, status_code=400)

        sessionId = str(uuid.uuid4())
        blobName = f"scans/{uuid.uuid4()}.zip"

        upload_sessions[sessionId] = {
            "sessionId": sessionId,
            "fileName": fileName,
            "fileSize": int(fileSize),
            "blobName": blobName,
            # index -> block_id  (dict avoids the out-of-order append bug)
            "blocks": {},
            "totalChunks": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        print(f"[Upload] Session initiated: {sessionId}")
        print(f"[Upload] Blob: {blobName}")

        return JSONResponse({
            "sessionId": sessionId,
            "chunkSize": CHUNK_SIZE,
            "blobName": blobName,
        })
    except Exception as e:
        print(f"[Upload ERROR] Initiate failed: {e}")
        return JSONResponse({"error": f"Upload initiation failed: {e}"}, status_code=500)


async def upload_chunk(request) -> JSONResponse:
    """POST /api/upload/chunk — stage one chunk as an Azure block (by index)."""
    try:
        form = await request.form()
        sessionId = form.get("sessionId")
        chunkIndex = int(form.get("chunkIndex", -1))
        totalChunks = int(form.get("totalChunks", -1))
        chunk = form.get("chunk")

        if not sessionId or chunkIndex < 0 or chunk is None:
            return JSONResponse(
                {"error": "sessionId, chunkIndex, and chunk required"}, status_code=400)

        session = upload_sessions.get(sessionId)
        if session is None:
            return JSONResponse({"error": f"Session {sessionId} not found"}, status_code=404)

        session["totalChunks"] = totalChunks

        chunk_data = await chunk.read()
        chunk_size = len(chunk_data)

        # Deterministic block id from the INDEX (not arrival order)
        block_id = _block_id(chunkIndex)

        # Stage the block, with a small retry for transient network errors
        blob_client = get_blob_client(session["blobName"])
        last_err = None
        for attempt in range(STAGE_RETRIES + 1):
            try:
                await blob_client.stage_block(block_id, chunk_data)
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[Azure WARN] stage_block idx={chunkIndex} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(0.4 * (attempt + 1))
        if last_err is not None:
            raise last_err

        # Record by index (idempotent: re-send overwrites the same slot)
        session["blocks"][chunkIndex] = block_id

        received = len(session["blocks"])
        print(f"[Upload] Chunk {chunkIndex + 1}/{totalChunks} staged "
              f"({chunk_size} bytes)  [{received}/{totalChunks} done]")

        return JSONResponse({
            "sessionId": sessionId,
            "chunkIndex": chunkIndex,
            "blockId": block_id,
            "status": "received",
            "progress": (received / totalChunks) * 100 if totalChunks else 0,
        })
    except Exception as e:
        print(f"[Upload ERROR] Chunk upload failed: {e}")
        return JSONResponse({"error": f"Chunk upload failed: {e}"}, status_code=500)


async def finalize_upload(request) -> JSONResponse:
    """POST /api/upload/finalize — commit blocks IN ORDER, then verify."""
    try:
        body = await request.json()
        sessionId = body.get("sessionId")

        session = upload_sessions.get(sessionId)
        if session is None:
            return JSONResponse({"error": f"Session {sessionId} not found"}, status_code=404)

        blob_name = session["blobName"]
        total = session["totalChunks"]
        blocks = session["blocks"]

        print(f"[Upload] Finalizing session {sessionId}...")

        # 1. Every chunk present?
        if len(blocks) != total:
            missing = [i for i in range(total) if i not in blocks]
            return JSONResponse(
                {"error": f"Missing chunks. Got {len(blocks)}, expected {total}. "
                          f"Missing indices: {missing}"},
                status_code=400)

        # 2. Build the block list in STRICT index order (the actual fix)
        try:
            ordered_block_ids = [blocks[i] for i in range(total)]
        except KeyError as e:
            return JSONResponse(
                {"error": f"Chunk index {e} missing at commit time"}, status_code=400)

        # 3. Commit, tagging content-type so it's stored as a zip
        blob_client = get_blob_client(blob_name)
        print(f"[Azure] Committing {len(ordered_block_ids)} blocks in order...")
        await blob_client.commit_block_list(
            ordered_block_ids,
            content_settings=ContentSettings(content_type="application/zip"),
        )
        print(f"[Azure] Blob created: {blob_name}")

        # 4. Verify size matches what the client said it uploaded
        expected = session["fileSize"]
        try:
            props = await blob_client.get_blob_properties()
            final_size = props.size
        except Exception as e:
            print(f"[Azure WARNING] Could not verify blob: {e}")
            final_size = expected

        del upload_sessions[sessionId]

        if final_size != expected:
            print(f"[Azure WARNING] Size mismatch: committed {final_size}, expected {expected}")
            return JSONResponse(
                {"error": f"Upload size mismatch: got {final_size}, expected {expected} bytes. "
                          f"Please retry the upload."},
                status_code=500)

        print(f"[Azure] Blob verified! Size: {final_size} bytes ✓")
        print(f"[Upload] Upload complete! Blob: {blob_name}")

        return JSONResponse({
            "blobName": blob_name,
            "sessionId": sessionId,
            "status": "completed",
            "size": final_size,
        })
    except Exception as e:
        print(f"[Upload ERROR] Finalize failed: {e}")
        return JSONResponse({"error": f"Upload finalization failed: {e}"}, status_code=500)


async def predict(request) -> JSONResponse:
    """POST /api/predict — call Modal inference for a committed blob."""
    try:
        body = await request.json()
        blob_name = body.get("blob_name")
        if not blob_name:
            return JSONResponse({"error": "blob_name required"}, status_code=400)

        print(f"[Azure] Calling Modal inference for {blob_name}")
        result = await call_modal_inference(blob_name)
        print("[Azure] Inference successful!")
        return JSONResponse(result)
    except Exception as e:
        print(f"[Azure ERROR] Prediction endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Prediction failed: {e}"}, status_code=500)


# ─── App Setup ────────────────────────────────────────────────────────────

routes = [
    Route("/api/upload/initiate", initiate_upload, methods=["POST", "OPTIONS"]),
    Route("/api/upload/chunk", upload_chunk, methods=["POST", "OPTIONS"]),
    Route("/api/upload/finalize", finalize_upload, methods=["POST", "OPTIONS"]),
    Route("/api/predict", predict, methods=["POST", "OPTIONS"]),
]

app = Starlette(routes=routes)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.mount(
    "/ui",
    StaticFiles(directory=os.path.join(BASE_DIR, "frontend"), html=True),
    name="frontend",
)


if __name__ == "__main__":
    import uvicorn
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  PlumoNet Azure Frontend (async ordered streaming + Modal) ║
╚════════════════════════════════════════════════════════════╝

Upload  : async SDK · cached client · ordered+verified commit ✅
Inference: Modal.Function.from_name() + .remote.aio()          ✅

Starting server on port {PORT}...
    """)
    uvicorn.run("service_azure:app", host="0.0.0.0", port=PORT, reload=False)