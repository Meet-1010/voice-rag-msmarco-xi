from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.routes import boot, router  # noqa: E402

WEB = ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index and warm the encoder before the port opens, so the first real
    # request is not the one that pays for it.
    boot()
    yield


app = FastAPI(title="Voice RAG over MSMARCO-XI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["api"]
    uvicorn.run(app, host=cfg["host"], port=cfg["port"])
