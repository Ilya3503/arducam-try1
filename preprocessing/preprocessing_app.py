import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
import open3d as o3d
import base64

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

def save_preview_projection(pcd, filename, projection="xy"):
    """Сохраняем PNG превью облака точек с помощью matplotlib"""
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
    plt.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


@app.get("/preprocessing/health_check", tags=["Служебные эндпоинты"], summary="Проверка активности сервиса препроцессинга")
async def get_preprocessing_health_check():
    return {"preprocessing": "working!"}


@app.get("/preview_before_preprocess", tags=["Служебные эндпоинты"], summary="Превью необработанного Point Cloud")
async def get_preview_before_preprocessing(
    projection: str = Query("xy", description="Проекция для превью: xy, xz или yz")
):
    if not INPUT_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Файл не создан, тут нечего смотреть, бро: {INPUT_FILE}")

    try:
        pcd = o3d.io.read_point_cloud(str(INPUT_FILE))
        save_preview_projection(pcd, PREVIEW_BEFORE_PRE_FILE, projection=projection)

        with open(PREVIEW_BEFORE_PRE_FILE, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {
        "status": "success",
        "file": PREVIEW_BEFORE_PRE_FILE.name,
        "projection": projection,
        "preview_base64": img_base64,
    }




# ------------------ Main Preprocess ------------------
@app.post("/preprocess", tags=["Эндпоинты обработки"], summary="Препроцессинг с PNG-превью (matplotlib)")
def preprocess(voxel_size: float = Query(0.01, description="Размер вокселя для фильтрации облака точек"),
               projection: str = Query("xy", description="Проекция для превью: xy, xz или yz")):
    if not INPUT_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {INPUT_FILE}")

    try:
        preprocess_point_cloud(str(INPUT_FILE), str(OUTPUT_FILE), voxel_size=voxel_size)

        pcd = o3d.io.read_point_cloud(str(OUTPUT_FILE))
        save_preview_projection(pcd, PREVIEW_FILE, projection=projection)

        with open(PREVIEW_FILE, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {
        "status": "success",
        "file": OUTPUT_FILE.name,
        "voxel_size": voxel_size,
        "projection": projection,
        "preview_base64": img_base64,
    }

