import os

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
import open3d as o3d
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from pre_functions import preprocess_point_cloud

from dotenv import load_dotenv

load_dotenv()

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
INPUT_FILE = SHARED_DIR / "last_pointcloud.ply"
OUTPUT_FILE = SHARED_DIR / "last_preprocessed_pointcloud.ply"
PREVIEW_FILE = SHARED_DIR / "last_preview.png"
PREVIEW_BEFORE_PRE_FILE = SHARED_DIR / "last_preview_before_pre.png"



app = FastAPI(
    title="Preprocessing Service",
    description="Сервис первичной обработки изображений формата Point Cloud",
    openapi_tags = [
        {"name": "Служебные эндпоинты"},
        {"name": "Эндпоинты обработки", "description": "Команды по обработке изображений"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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




def save_preview_projection(pcd, projection="xy") -> str:
    points = np.asarray(pcd.points)

    if points.shape[0] == 0:
        raise ValueError("PointCloud пуст, нечего визуализировать")

    if projection == "xy":
        x, y = points[:, 0], points[:, 1]
    elif projection == "xz":
        x, y = points[:, 0], points[:, 2]
    elif projection == "yz":
        x, y = points[:, 1], points[:, 2]
    else:
        raise ValueError(f"Неизвестная проекция: {projection}")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, c="black")
    plt.axis("equal")
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64



@app.get("/preprocessing/health_check", tags=["Служебные эндпоинты"], summary="Проверка активности сервиса препроцессинга")
async def get_preprocessing_health_check():
    return {"preprocessing": "working!"}


@app.get("/preview_before_preprocess", tags=["Служебные эндпоинты"], summary="Превью необработанного Point Cloud")
def get_preview_before_preprocessing(
    use_latest: bool = Query(True, description="Брать последний файл или нет"),
    folder: str = Query(None, description="Имя подпапки, если use_latest=False"),
    filename: str = Query(None, description="Имя файла, если use_latest=False"),
    projection: str = Query("xy", description="Проекция для превью: xy, xz или yz")
):
    try:
        input_file, _ = get_input_file(use_latest, folder, filename)

        # Загружаем облако точек
        pcd = o3d.io.read_point_cloud(str(input_file))

        # Получаем превью в base64 (без записи на диск)
        img_base64 = save_preview_projection(pcd, projection=projection)

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации превью: {exc}")

    return {
        "status": "success",
        "file": input_file.name,
        "folder": str(input_file.parent),
        "projection": projection,
        "preview_base64": img_base64,
    }





# ------------------ Main Preprocess ------------------
@app.post("/preprocess", tags=["Эндпоинты обработки"], summary="Препроцессинг с PNG-превью (matplotlib)")
def preprocess(
    use_latest: bool = Query(True, description="Брать последний файл или нет"),
    folder: str = Query(None, description="Имя подпапки, если use_latest=False"),
    filename: str = Query(None, description="Имя файла, если use_latest=False"),
    projection: str = Query("xy", description="Проекция для превью: xy, xz или yz")
):
    try:
        input_file, output_file = get_input_file(use_latest, folder, filename)

        preprocess_point_cloud(str(input_file), str(output_file))

        pcd = o3d.io.read_point_cloud(str(output_file))

        img_base64 = save_preview_projection(pcd, projection=projection)

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {
        "status": "success",
        "file": output_file.name,
        "folder": str(output_file.parent),
        "projection": projection,
        "preview_base64": img_base64,
    }

