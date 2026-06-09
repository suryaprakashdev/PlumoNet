from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Starlette()

app.mount(
    "/",
    StaticFiles(
        directory=os.path.join(BASE_DIR, "frontend"),
        html=True
    ),
    name="frontend"
)