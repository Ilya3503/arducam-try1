import os
import io
import base64
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

SHARED_DIR = os.environ.get("SHARED_DIR")
if not SHARED_DIR:
    raise RuntimeError("Environment variable SHARED_DIR is not set")
SHARED_DIR = Path(SHARED_DIR)
if not SHARED_DIR.exists():
    raise FileNotFoundError(f"SHARED_DIR does not exist: {SHARED_DIR}")

CAMERA_URL = os.environ.get("CAMERA_URL")
CAMERA_URL_INFO = os.environ.get("CAMERA_URL_INFO")

app = FastAPI(
    title="Pipeline Service",
    description="Основной сервис системы",
    openapi_tags = [
        {"name": "Служебные эндпоинты"},
        {"name": "Эндпоинты обработки", "description": "Команды по обработке изображений и CV"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/pipeline/health_check", tags=["Служебные эндпоинты"], summary="Проверка активности pipeline-сервиса")
async def health():
    return {"pipeline": "working!"}

def get_latest_capture_dir():
    shared = Path(SHARED_DIR)
    if not shared.exists():
        raise FileNotFoundError(f"Папка SHARED_DIR не найдена: {shared}")
    subdirs = [d for d in shared.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"В {shared} нет поддиректорий с результатами")
    latest = max(subdirs, key=lambda d: d.name)
    return latest



def depth_to_base64_png(depth_path) -> str:
    depth_path = str(depth_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Файл {depth_path} не найден")

    depth = np.load(depth_path)
    if depth.ndim != 2:
        raise ValueError("Ожидалась 2D depth-карта")

    vmax = np.percentile(depth[depth > 0], 95) if np.any(depth > 0) else 1.0
    norm = np.clip(depth / vmax, 0, 1)
    img_arr = (255 * (1.0 - norm)).astype(np.uint8)

    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/process", tags=["Эндпоинты обработки"], summary="Запуск создания изображения и получения превью")
async def process():
    try:
        r = requests.get(CAMERA_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка запроса к камере: {e}")

    latest_capture_dir = get_latest_capture_dir()
    depth_file = latest_capture_dir / f"{latest_capture_dir.name}_depth.npy"
    ply_file = latest_capture_dir / f"{latest_capture_dir.name}_pointcloud.ply"

    try:
        preview_b64 = depth_to_base64_png(depth_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке depth map: {e}")

    return {
        "status": "Успешно",
        "preview_b64": preview_b64,
        "ply_file": str(ply_file),
    }



@app.get("/camera/info", tags=["Эндпоинты обработки"], summary="Получение информации о камере")
async def get_camera_info_endpoint_pipeline():
    try:
        r = requests.get(CAMERA_URL_INFO, timeout=10)
        r.raise_for_status()
        camera_info = r.json()
    except Exception as e:
        return {"status": "Ошибка получения информации о камере", "message": str(e)}

    return {"status": "Успешно", "info": camera_info}




@app.get("/files")
async def get_files_list():
    shared_path = Path(SHARED_DIR)
    if not shared_path.exists():
        return {"status": "Ошибка", "message": f"Папка {SHARED_DIR} не найдена"}

    capture_dirs = [d for d in shared_path.iterdir() if d.is_dir()]
    if not capture_dirs:
        return {"status": "Папка успешно прочитана", "message": "Подпапок с результатами нет"}

    all_files = {}
    for d in sorted(capture_dirs):
        files_in_dir = [f.name for f in d.iterdir() if f.is_file()]
        all_files[d.name] = files_in_dir

    return {"status": "Папка успешно прочитана", "capture_dirs": all_files}





# @app.get("/download_ply", tags=["Эндпоинты обработки"], summary="Скачивание основного файла PointCloud")
# async def download_ply(name: str):
#     safe_name = os.path.basename(name)
#     path = os.path.join(SHARED_DIR, safe_name)
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="PLY не найден")
#     return FileResponse(path, media_type="application/octet-stream", filename=safe_name)
