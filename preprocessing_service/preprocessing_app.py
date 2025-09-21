import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException

from pre_functions import preprocess_point_cloud

from dotenv import load_dotenv

load_dotenv()

SHARED_DIR = Path(os.getenv("SHARED_DIR", "/shared"))
INPUT_FILE = SHARED_DIR / "last_pointcloud.ply"
OUTPUT_FILE = SHARED_DIR / "last_preprocessed_pointcloud.ply"



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


@app.post("/preprocess", tags=["Эндпоинты обработки"], summary="Препроцессинг (первичная обработка) данных от камеры")
def preprocess():
    if not INPUT_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {INPUT_FILE}")

    try:
        preprocess_point_cloud(str(INPUT_FILE), str(OUTPUT_FILE))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка препроцессинга данных: {exc}")

    return {"status": "success", "file": OUTPUT_FILE.name}