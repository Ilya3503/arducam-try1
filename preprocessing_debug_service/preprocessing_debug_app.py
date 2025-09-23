import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import open3d as o3d

from pre_functions_debug import load_point_cloud, save_point_cloud, voxel_downsample

# -------------------- Настройки --------------------
SHARED_DIR = os.environ.get("SHARED_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared"))
PROCESSED_DIR = Path(SHARED_DIR) / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Preprocessing Debug Service",
    description="Сервис для тестирования функций препроцессинга",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Служебные эндпоинты --------------------
@app.get("/files", summary="Список доступных файлов для обработки")
def list_files():
    if not SHARED_DIR.exists():
        raise HTTPException(status_code=404, detail="shared/ not found")

    files = [f.name for f in SHARED_DIR.iterdir() if f.suffix.lower() in [".ply", ".npy"]]
    return {"files": files}


# -------------------- Обработка: Voxel Downsample --------------------
@app.post("/preprocess/voxel_downsample", summary="Вокселизация Point Cloud")
def preprocess_voxel_downsample(
    filename: str = Query(..., description="Имя входного файла из /files"),
    voxel_size: float = Query(0.01, description="Размер вокселя (м)")
):
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        # Загружаем
        pcd = load_point_cloud(str(input_path))
        # Обрабатываем
        pcd_down = voxel_downsample(pcd, voxel_size=voxel_size)

        # Сохраняем с меткой времени
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{timestamp}_voxel_{voxel_size:.3f}.ply"
        out_path = PROCESSED_DIR / out_name
        save_point_cloud(pcd_down, str(out_path))

        # Автоматическая визуализация (локально, если есть GUI)
        try:
            o3d.visualization.draw_geometries([pcd_down])
        except Exception as vis_err:
            # если GUI нет (например, на Pi без экрана), просто логируем
            print(f"Визуализация недоступна: {vis_err}")

        return {
            "status": "ok",
            "input_file": filename,
            "output_file": out_name,
            "points_in": len(pcd.points),
            "points_out": len(pcd_down.points),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
