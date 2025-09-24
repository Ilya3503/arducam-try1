import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import open3d as o3d
import base64

from pre_functions import preprocess_point_cloud

from dotenv import load_dotenv

load_dotenv()

SHARED_DIR = Path(os.environ.get("SHARED_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared")))
INPUT_FILE = os.path.join(SHARED_DIR, "last_pointcloud.ply")
OUTPUT_FILE = os.path.join(SHARED_DIR, "last_preprocessed_pointcloud.ply")
PREVIEW_FILE = os.path.join(SHARED_DIR, "last_preview.png")



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



# ------------------ Main Preprocess ------------------
@app.post("/preprocess", tags=["Эндпоинты обработки"], summary="Препроцессинг с PNG-превью (offscreen)")
def preprocess():
    if not INPUT_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {INPUT_FILE}")

    try:
        preprocess_point_cloud(str(INPUT_FILE), str(OUTPUT_FILE))

        pcd = o3d.io.read_point_cloud(str(OUTPUT_FILE))

        width, height = 800, 600
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        render.scene.add_geometry("pcd", pcd, mat)
        render.setup_camera(60.0, pcd.get_axis_aligned_bounding_box(), pcd.get_center())

        img_o3d = render.render_to_image()
        o3d.io.write_image(str(PREVIEW_FILE), img_o3d)

        with open(PREVIEW_FILE, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {
        "status": "success",
        "file": OUTPUT_FILE.name,
        "preview_base64": img_base64,
    }
