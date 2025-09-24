import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException

os.environ["OPEN3D_CPU_RENDERING"] = "true"

import open3d as o3d
import base64

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




@app.get("/preprocessing/health_check", tags=["Служебные эндпоинты"], summary="Проверка активности сервиса препроцессинга")
async def get_preprocessing_health_check():
    return {"preprocessing": "working!"}


@app.get("/preview_before_preprocess", tags=["Служебные эндпоинты"], summary="Превью необработанного Point Cloud")
async def get_preview_before_preprocessing():
    if not INPUT_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Файл не создан, тут нечего смотреть, бро: {INPUT_FILE}")

    try:
        pcd = o3d.io.read_point_cloud(str(INPUT_FILE))

        width, height = 800, 600
        try:
            render = o3d.visualization.rendering.OffscreenRenderer(width, height)
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            render.scene.add_geometry("pcd", pcd, mat)
            render.setup_camera(60.0, pcd.get_axis_aligned_bounding_box(), pcd.get_center())

            img_o3d = render.render_to_image()
            o3d.io.write_image(str(PREVIEW_BEFORE_PRE_FILE), img_o3d)

            with open(PREVIEW_BEFORE_PRE_FILE, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            render.release()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {
        "status": "success",
        "file": PREVIEW_BEFORE_PRE_FILE.name,
        "preview_base64": img_base64,
    }



# ------------------ Main Preprocess ------------------
@app.get("/preprocess", tags=["Эндпоинты обработки"], summary="Препроцессинг с PNG-превью")
def preprocess():
    if not INPUT_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {INPUT_FILE}")

    try:
        # Выполняем только вокселизацию
        preprocess_point_cloud(str(INPUT_FILE), str(OUTPUT_FILE))

        pcd = o3d.io.read_point_cloud(str(OUTPUT_FILE))

        width, height = 800, 600
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        try:
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            render.scene.add_geometry("pcd", pcd, mat)
            render.setup_camera(60.0, pcd.get_axis_aligned_bounding_box(), pcd.get_center())

            img_o3d = render.render_to_image()
            o3d.io.write_image(str(PREVIEW_FILE), img_o3d)

            with open(PREVIEW_FILE, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            render.release()  # Освобождаем ресурсы
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {
        "status": "success",
        "file": OUTPUT_FILE.name,
        "preview_base64": img_base64,
    }
