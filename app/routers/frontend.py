
from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.config import FRONTEND_DIR

router = APIRouter(tags=["Frontend"])


@router.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
