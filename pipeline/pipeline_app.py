import os
import io
import base64
import numpy as np
from pathlib import Path
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv




load_dotenv()

SHARED_DIR = os.environ.get("SHARED_DIR")
if not SHARED_DIR:
    raise RuntimeError("Environment variable SHARED_DIR is not set")
SHARED_DIR = Path(SHARED_DIR)
if not SHARED_DIR.exists():
    raise FileNotFoundError(f"SHARED_DIR does not exist: {SHARED_DIR}")

CAMERA_URL = os.environ.get("CAMERA_URL")
CAMERA_URL_INFO = os.environ.get("CAMERA_URL_INFO")
PROCESSING_URL = os.environ.get("PROCESSING_URL", "http://100.96.67.98:8006/process")

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

def get_input_file(use_latest: bool = True, folder: str = None, filename: str = None):
    shared = Path(SHARED_DIR)
    if not shared.exists():
        raise FileNotFoundError(f"SHARED_DIR не найден: {shared}")

    if use_latest:
        subdirs = [d for d in shared.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError("Нет подпапок с результатами")
        latest_dir = max(subdirs, key=lambda d: d.name)
        latest_pointcloud_file = latest_dir / f"{latest_dir.name}_pointcloud.ply"
        if not latest_pointcloud_file.exists():
            raise FileNotFoundError(f"PLY-файл не найден в последней папке: {latest_pointcloud_file}")
        input_file = latest_pointcloud_file
        output_file = latest_dir / f"{latest_dir.name}_preprocessed.ply"
    else:
        if not folder or not filename:
            raise ValueError("Если use_latest=False, нужно передать folder и filename")
        folder_path = shared / folder
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")
        input_file = folder_path / filename
        if not input_file.exists():
            raise FileNotFoundError(f"Файл не найден: {input_file}")
        output_file = folder_path / f"{input_file.stem}_preprocessed.ply"

    return input_file, output_file

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


@app.post("/model_process", tags=["Эндпоинты обработки"], summary="Отправка PLY-файла в ML-сервис Processing")
def model_process(
    use_latest: bool = Query(True, description="Брать последний файл или нет"),
    folder: str = Query(None, description="Имя подпапки, если use_latest=False"),
    filename: str = Query(None, description="Имя файла, если use_latest=False")
):
    try:
        input_file, _ = get_input_file(use_latest, folder, filename)
        if not input_file.exists():
            raise FileNotFoundError(f"Файл не найден: {input_file}")

        with open(input_file, "rb") as f:
            files = {"file": (input_file.name, f, "application/octet-stream")}
            response = requests.post(PROCESSING_URL, files=files, timeout=30)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            raise HTTPException(status_code=response.status_code, detail=f"Processing error: {response.text}")

        try:
            resp_json = response.json()
        except ValueError:
            resp_json = {"raw_response_text": response.text}

        return {"status": "success", "processing_response": resp_json}

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=502, detail=f"Ошибка соединения с Processing: {req_err}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка при отправке файла: {exc}")







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
