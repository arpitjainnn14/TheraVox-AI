"""HTML page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/text", response_class=HTMLResponse)
async def text_page(request: Request):
    """Text analysis page."""
    return templates.TemplateResponse("text.html", {"request": request})


@router.get("/audio", response_class=HTMLResponse)
async def audio_page(request: Request):
    """Audio analysis page."""
    return templates.TemplateResponse("audio.html", {"request": request})


@router.get("/vision", response_class=HTMLResponse)
async def vision_page(request: Request):
    """Vision analysis page."""
    return templates.TemplateResponse("vision.html", {"request": request})


@router.get("/developers", response_class=HTMLResponse)
async def developers_page(request: Request):
    """Developers page."""
    return templates.TemplateResponse("developers.html", {"request": request})


@router.get("/wellness", response_class=HTMLResponse)
async def wellness_page(request: Request):
    """Wellness page."""
    return templates.TemplateResponse("wellness.html", {"request": request})
