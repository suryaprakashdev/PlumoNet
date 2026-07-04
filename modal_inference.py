"""
PlumoNet Modal GPU Inference — Pattern 5 (GPU Streaming Endpoint)
==================================================================

Hosts the 3D U-Net segmentation + 3D ResNet-10 classification pipeline as a
single Modal-hosted HTTPS endpoint that streams progress via Server-Sent
Events (SSE) while it runs, instead of blocking until the whole ~2-3 minute
inference finishes.

Warm loading: `InferencePipeline3D` is constructed once inside `web_app()`
(the `@modal.asgi_app()` factory), which Modal calls once per container boot,
not once per request — so a container that serves back-to-back requests
within `scaledown_window` of each other reuses the already-loaded ONNX
sessions instead of reloading them every time.

Access control: this endpoint is a public HTTPS URL once deployed. It is
called ONLY by service_azure.py (server-to-server), never by the browser
directly, and is gated by a shared-secret header checked before any GPU work
starts (see `_check_auth`). CORS is intentionally not configured here since
the caller is a server, not a browser — CORS provides no protection against
a direct curl/script call, only the shared secret does.

One-time setup before deploying:
    modal secret create azure-connection-string AZURE_STORAGE_CONNECTION_STRING="..."
    modal secret create plumonet-shared-secret PLUMONET_SHARED_SECRET="<random-token>"
    # Set the same PLUMONET_SHARED_SECRET value as an env var on the Azure
    # Container App (service_azure.py reads it to sign its request to Modal).

Deploy with:
    modal deploy modal_inference.py   # run from repo root — add_local_file
                                       # paths below are relative to it.
"""

import asyncio
import base64
import os
import queue as queue_mod
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List

import modal

# ─── Modal App ─────────────────────────────────────────────────────────────

app = modal.App("plumonet-inference")

# ─── GPU Image ─────────────────────────────────────────────────────────────
# Notes:
#  - Base image: plain debian_slim + `pip install onnxruntime-gpu` does NOT
#    include the CUDA runtime libraries onnxruntime-gpu needs at import time
#    (libcublasLt.so.11 etc.) — CUDAExecutionProvider then fails to load and
#    onnxruntime silently falls back to CPUExecutionProvider, so inference
#    runs on CPU despite the A100 being attached. Using an official NVIDIA
#    CUDA runtime image (matching onnxruntime-gpu==1.17.1's CUDA 11.8/cuDNN 8
#    build) ships those libraries so the CUDA EP actually loads.
#  - apt: libgl1/libglib2.0-0 satisfy transitive OpenGL deps (monai/skimage/mpl).
#  - Model code + ONNX weights are baked into the image via add_local_*.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements_modal.txt")
    .pip_install("onnxruntime-gpu==1.17.1")
    .add_local_file("inference_3d.py", "/root/inference_3d.py")
    .add_local_file("unet3d.py", "/root/unet3d.py")
    .add_local_file("resnet3d.py", "/root/resnet3d.py")
    .add_local_file("checkpoints/unet3d.onnx", "/root/checkpoints/unet3d.onnx")
    .add_local_file("checkpoints/resnet3d.onnx", "/root/checkpoints/resnet3d.onnx")
)

# ─── Helpers (ported from the previous sync implementation) ────────────────

_SENTINEL = object()


def generate_annotated_slice(arr2d, mask2d=None, candidates=None, vmin=-1000, vmax=400):
    """
    Render a 2D HU slice with OpenCV: HU windowing -> 8-bit -> light smoothing,
    then draw a red crosshair + label for each malignant candidate (prob >= 0.5).
    Returns a base64-encoded PNG string.
    """
    import numpy as np
    import cv2

    normed = np.clip((arr2d.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    img_8u = (normed * 255).astype(np.uint8)
    img_smooth = cv2.GaussianBlur(img_8u, (3, 3), 0)
    img_bgr = cv2.cvtColor(img_smooth, cv2.COLOR_GRAY2BGR)

    RED = (0, 0, 255)  # BGR
    for cand in (candidates or []):
        if cand["probability"] < 0.5:
            continue
        prob = cand["probability"]
        cy, cx = cand["center_2d"]
        cv2.drawMarker(img_bgr, (cx, cy), RED, cv2.MARKER_CROSS, 10, 2)
        label = f"#{cand['candidate_index']} ({prob:.0%})"
        cv2.putText(img_bgr, label, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)

    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode()


def find_dicom_root(base_dir: str) -> str:
    """Walk extracted ZIP; return first dir containing DICOM files (skip __MACOSX)."""
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in ("gradcam_out", "__MACOSX")]
        if dirpath.endswith("__MACOSX"):
            continue
        if any(f.lower().endswith('.dcm') or ('.' not in f and not f.startswith('._'))
               for f in filenames):
            return dirpath
    return base_dir


def _run_pipeline_sync(pipeline, dicom_root: str, event_queue: "queue_mod.Queue"):
    """
    Runs on a background thread: drains pipeline.run_volume_streaming()'s sync
    generator and pushes each event onto a queue the async SSE handler drains.
    This is the bridge that lets a synchronous, GPU-bound pipeline report
    progress to an async endpoint without needing any cross-process streaming
    API — everything here happens inside one container/process.
    """
    try:
        for event in pipeline.run_volume_streaming(
            dicom_root, aggregation="top_k", k=5, min_malignancy_prob=0.0
        ):
            event_queue.put(event)
    except Exception as exc:
        event_queue.put({"stage": "error", "message": str(exc)})
    finally:
        event_queue.put(_SENTINEL)


def _sse(payload: Dict[str, Any]) -> bytes:
    import json
    return f"data: {json.dumps(payload)}\n\n".encode()


# ─── Streaming GPU Endpoint ─────────────────────────────────────────────


@app.function(
    image=image,
    gpu="a100",
    secrets=[
        modal.Secret.from_name("azure-connection-string"),
        modal.Secret.from_name("plumonet-shared-secret"),
    ],
    timeout=600,
    min_containers=0,       # true scale-to-zero — traffic is low/sporadic
    max_containers=3,       # cost/blast-radius safety rail, not a throughput target
    scaledown_window=300,   # keep a container warm 5 min after its last request
                            # so back-to-back scans reuse the already-loaded model
)
@modal.asgi_app()
def web_app():
    """
    Called once per container boot. Loading InferencePipeline3D here (not
    inside the route handler) is what makes this warm: the container reuses
    this same pipeline instance for every request it serves until Modal
    reclaims it after `scaledown_window` seconds of idling.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from azure.storage.blob import BlobServiceClient

    from inference_3d import InferencePipeline3D

    print("[Cache] Loading InferencePipeline3D (container boot)...")
    pipeline = InferencePipeline3D(
        unet_onnx_path="/root/checkpoints/unet3d.onnx",
        resnet_onnx_path="/root/checkpoints/resnet3d.onnx",
    )
    print("[Cache] Pipeline loaded — reused for every request on this container.")

    api = FastAPI(title="PlumoNet GPU Inference (streaming)")

    def _check_auth(request: Request) -> bool:
        expected = os.environ.get("PLUMONET_SHARED_SECRET")
        provided = request.headers.get("x-plumonet-secret")
        return bool(expected) and provided == expected

    @api.post("/stream")
    async def stream(request: Request):
        if not _check_auth(request):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        blob_name = body.get("blob_name")
        if not blob_name:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "blob_name required"}, status_code=400)

        return StreamingResponse(
            _event_stream(blob_name, pipeline),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _event_stream(blob_name: str, pipeline):
        print(f"[Modal GPU] Starting streaming inference for: {blob_name}")
        try:
            connection_string = (
                os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                or os.environ.get("MODAL_AZURE_STORAGE_CONNECTION_STRING")
            )
            if not connection_string:
                yield _sse({"stage": "error", "message": "AZURE_STORAGE_CONNECTION_STRING not found"})
                return

            container = "dicom-uploads"
            bsc = BlobServiceClient.from_connection_string(connection_string)

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                blob_client = bsc.get_blob_client(container=container, blob=blob_name)

                t0 = time.perf_counter()
                data = await asyncio.to_thread(
                    lambda: blob_client.download_blob().readall()
                )
                print(f"[Timing] download: {time.perf_counter() - t0:.1f}s ({len(data)} bytes)")
                if not data.startswith(b"PK"):
                    yield _sse({"stage": "error", "message": "downloaded blob is not a ZIP"})
                    return
                with open(zip_path, "wb") as f:
                    f.write(data)
                yield _sse({"stage": "downloaded", "bytes": len(data)})

                t0 = time.perf_counter()
                await asyncio.to_thread(shutil.unpack_archive, zip_path, tmpdir, "zip")
                dicom_root = find_dicom_root(tmpdir)
                print(f"[Timing] unzip+find_root: {time.perf_counter() - t0:.1f}s")

                # ── Run the pipeline on a background thread, draining its
                #    progress events onto the event loop as they arrive ──
                t0 = time.perf_counter()
                event_queue: "queue_mod.Queue" = queue_mod.Queue()
                threading.Thread(
                    target=_run_pipeline_sync,
                    args=(pipeline, dicom_root, event_queue),
                    daemon=True,
                ).start()

                result = None
                while True:
                    event = await asyncio.to_thread(event_queue.get)
                    if event is _SENTINEL:
                        break
                    if event.get("stage") == "result":
                        result = event["result"]
                        continue
                    if event.get("stage") == "error":
                        yield _sse(event)
                        return
                    yield _sse(event)
                print(f"[Timing] pipeline (wall, incl. thread handoff): {time.perf_counter() - t0:.1f}s")

                if result is None:
                    yield _sse({"stage": "error", "message": "pipeline ended without a result"})
                    return

                # ── Build the response payload, streaming render progress ──
                seg_mask = getattr(result, "segmentation_mask", None)
                candidates = result.candidates
                vol = result.volume_iso

                patient_score = max((c.probability for c in candidates), default=0.0)
                patient_prediction = "Malignant" if patient_score >= 0.5 else "Benign"

                def get_candidates_in_slice(plane: str, idx: int) -> List[dict]:
                    in_slice = []
                    for c in candidates:
                        (z0, z1), (y0, y1), (x0, x1) = c.bbox
                        if plane == "axial" and z0 <= idx <= z1:
                            in_slice.append({
                                "candidate_index": int(c.candidate_index),
                                "probability": float(c.probability),
                                "center_2d": (int(c.centroid[1]), int(c.centroid[2])),
                            })
                        elif plane == "coronal" and y0 <= idx <= y1:
                            in_slice.append({
                                "candidate_index": int(c.candidate_index),
                                "probability": float(c.probability),
                                "center_2d": (int(c.centroid[0]), int(c.centroid[2])),
                            })
                        elif plane == "sagittal" and x0 <= idx <= x1:
                            in_slice.append({
                                "candidate_index": int(c.candidate_index),
                                "probability": float(c.probability),
                                "center_2d": (int(c.centroid[0]), int(c.centroid[1])),
                            })
                    return in_slice

                def render_plane(plane: str):
                    views = {}
                    for idx in result.active_slices[plane]:
                        cands = get_candidates_in_slice(plane, idx)
                        if plane == "axial":
                            arr, m = vol[idx, :, :], seg_mask[idx, :, :] if seg_mask is not None else None
                        elif plane == "coronal":
                            arr, m = vol[:, idx, :], seg_mask[:, idx, :] if seg_mask is not None else None
                        else:
                            arr, m = vol[:, :, idx], seg_mask[:, :, idx] if seg_mask is not None else None
                        views[str(idx)] = {"image": generate_annotated_slice(arr, m, cands), "nodules": cands}
                    return views

                candidate_views: Dict[str, Dict[str, Any]] = {}
                planes = ["axial", "coronal", "sagittal"]
                t0 = time.perf_counter()
                for i, plane in enumerate(planes):
                    n_slices = len(result.active_slices[plane])
                    t_plane = time.perf_counter()
                    candidate_views[plane] = await asyncio.to_thread(render_plane, plane)
                    print(f"[Timing] render {plane}: {time.perf_counter() - t_plane:.1f}s ({n_slices} slices)")
                    yield _sse({"stage": "rendering", "current": i + 1, "total": len(planes)})
                print(f"[Timing] rendering total: {time.perf_counter() - t0:.1f}s")

                payload = {
                    "patient_score": float(patient_score),
                    "prediction": str(patient_prediction),
                    "num_candidates": int(len(candidates)),
                    "top_candidates": [
                        {
                            "centroid": [int(v) for v in c.centroid],
                            "prob": float(c.probability),
                            "prediction": str(c.prediction),
                            "volume_voxels": int(c.volume_voxels),
                            "candidate_index": int(c.candidate_index),
                        }
                        for c in candidates
                    ],
                    "candidate_views": candidate_views,
                    "metadata": {
                        "patient_id": str(result.metadata.get("patient_id", "")),
                        "study_date": str(result.metadata.get("study_date", "")),
                        "series_uid": str(result.metadata.get("series_uid", "")),
                        "volume_shape": [int(v) for v in result.metadata.get("volume_shape", [])],
                        "total_time_s": round(float(result.total_time_ms) / 1000, 1),
                        "seg_time_s": round(float(result.seg_time_ms) / 1000, 1),
                        "cls_time_s": round(float(result.cls_time_ms) / 1000, 1),
                    },
                }
                print(f"[Modal GPU] Inference complete ✓  prediction={payload['prediction']} "
                      f"candidates={payload['num_candidates']}")
                yield _sse({"stage": "done", "result": payload})

        except Exception as exc:
            print(f"[Modal GPU] ERROR: {exc}")
            yield _sse({"stage": "error", "message": str(exc)})

    return api


if __name__ == "__main__":
    print("Deploy with:  modal deploy modal_inference.py  (run from repo root)")
