import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile

import open3d as o3d

from pre_functions_debug import load_point_cloud_noresize, save_point_cloud, voxel_downsample, remove_noise, remove_plane, crop_roi, load_point_cloud


# -------------------- Настройки --------------------
SHARED_DIR = Path(os.environ.get("SHARED_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared")))
PROCESSED_DIR = SHARED_DIR


app = FastAPI(
    title="Preprocessing Debug Service",
    description="Сервис для тестирования функций препроцессинга",
    openapi_tags=[
        {"name": "Служебные эндпоинты"},
        {"name": "Эндпоинты обработки"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Служебные эндпоинты --------------------
@app.get("/files", tags=["Служебные эндпоинты"], summary="Список доступных файлов для обработки")
def list_files():
    if not SHARED_DIR.exists():
        raise HTTPException(status_code=404, detail="shared/ not found")

    files = [f.name for f in SHARED_DIR.iterdir() if f.suffix.lower() in [".ply", ".npy"]]
    return {"files": files}

# -------------------- Обработка: Voxel Downsample --------------------
@app.post("/preprocess/voxel_downsample", tags=["Эндпоинты обработки"], summary="Вокселизация Point Cloud")
def preprocess_voxel_downsample(
    filename: str = Query(..., description="Имя входного файла из /files"),
    voxel_size: float = Query(0.01, description="Размер вокселя (м)")
):
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        # Загружаем
        pcd = load_point_cloud_noresize(str(input_path))
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



# -------------------- Обработка: Remove Noise --------------------
@app.post("/preprocess/remove_noise", tags=["Эндпоинты обработки"], summary="Удаление шума (Statistical Outlier Removal)")
def preprocess_remove_noise(
    filename: str = Query(..., description="Имя входного файла из /files"),
    nb_neighbors: int = Query(20, description="Количество соседей для оценки"),
    std_ratio: float = Query(2.0, description="Пороговое значение стандартного отклонения")
):
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        pcd = load_point_cloud_noresize(str(input_path))
        pcd_clean = remove_noise(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{timestamp}_denoised_{nb_neighbors}_{std_ratio}.ply"
        out_path = PROCESSED_DIR / out_name
        save_point_cloud(pcd_clean, str(out_path))

        try:
            o3d.visualization.draw_geometries([pcd_clean])
        except Exception as vis_err:
            print(f"Визуализация недоступна: {vis_err}")

        return {
            "status": "ok",
            "input_file": filename,
            "output_file": out_name,
            "points_in": len(pcd.points),
            "points_out": len(pcd_clean.points),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Обработка: Remove Plane --------------------
@app.post("/preprocess/remove_plane", tags=["Эндпоинты обработки"], summary="Удаление плоскости (RANSAC plane segmentation)")
def preprocess_remove_plane(
    filename: str = Query(..., description="Имя входного файла из /files"),
    distance_threshold: float = Query(0.01, description="Максимальная дистанция до плоскости"),
    ransac_n: int = Query(3, description="Количество точек для оценки плоскости"),
    num_iterations: int = Query(1000, description="Количество итераций RANSAC")
):
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        pcd = load_point_cloud_noresize(str(input_path))
        pcd_no_plane = remove_plane(pcd, distance_threshold=distance_threshold,
                                    ransac_n=ransac_n, num_iterations=num_iterations)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{timestamp}_plane_removed_{distance_threshold:.6f}.ply"
        out_path = PROCESSED_DIR / out_name
        save_point_cloud(pcd_no_plane, str(out_path))

        try:
            o3d.visualization.draw_geometries([pcd_no_plane])
        except Exception as vis_err:
            print(f"Визуализация недоступна: {vis_err}")

        return {
            "status": "ok",
            "input_file": filename,
            "output_file": out_name,
            "points_in": len(pcd.points),
            "points_out": len(pcd_no_plane.points),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Обработка: Crop ROI --------------------
@app.post("/preprocess/crop_roi", tags=["Эндпоинты обработки"], summary="Обрезка области интереса (ROI)")
def preprocess_crop_roi(
    filename: str = Query(..., description="Имя входного файла из /files"),
    min_x: float = Query(..., description="Минимальное X"),
    min_y: float = Query(..., description="Минимальное Y"),
    min_z: float = Query(..., description="Минимальное Z"),
    max_x: float = Query(..., description="Максимальное X"),
    max_y: float = Query(..., description="Максимальное Y"),
    max_z: float = Query(..., description="Максимальное Z"),
):
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        pcd = load_point_cloud_noresize(str(input_path))
        pcd_cropped = crop_roi(pcd, min_bound=(min_x, min_y, min_z), max_bound=(max_x, max_y, max_z))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{timestamp}_cropped.ply"
        out_path = PROCESSED_DIR / out_name
        save_point_cloud(pcd_cropped, str(out_path))

        try:
            o3d.visualization.draw_geometries([pcd_cropped])
        except Exception as vis_err:
            print(f"Визуализация недоступна: {vis_err}")

        return {
            "status": "ok",
            "input_file": filename,
            "output_file": out_name,
            "points_in": len(pcd.points),
            "points_out": len(pcd_cropped.points),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# -------------------- Просмотр Point Cloud --------------------
@app.post("/preprocess/view_cloud", tags=["Эндпоинты обработки"], summary="Просмотр облака точек")
def preprocess_view_cloud(
    filename: str = Query(..., description="Имя входного файла из /files"),
    show_axes: bool = Query(False, description="Показать оси координат"),
    divisions: int = Query(10, description="Количество делений на каждой оси")
):
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        pcd = load_point_cloud_noresize(str(input_path))
        geometries = [pcd]

        if show_axes:
            # Диапазоны облака точек
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            center = (min_bound + max_bound) / 2
            ranges = max_bound - min_bound


        try:
            o3d.visualization.draw_geometries(geometries)
        except Exception as vis_err:
            print(f"Визуализация недоступна: {vis_err}")

        return {
            "status": "ok",
            "input_file": filename,
            "points": len(pcd.points),
            "show_axes": show_axes,
            "divisions": divisions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/preprocess/analyze_cloud", tags=["Эндпоинты обработки"], summary="Анализ и визуализация облака точек")
def preprocess_analyze_cloud(
    filename: str = Query(..., description="Имя входного файла из /shared")
):
    """
    Анализирует облако точек: считает количество точек, диапазон по осям,
    центр и визуализирует облако с координатной осью.
    """
    input_path = SHARED_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден в shared/")

    try:
        # Загрузка облака
        pcd = load_point_cloud_noresize(str(input_path))
        first_5_points = np.asarray(pcd.points[:5])

        # Преобразование в numpy для анализа
        points_count = len(pcd.points)
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()

        # Создание координатной системы (размер оси подбирается автоматически)
        axis_size = max(extent) * 0.1 if np.all(extent > 0) else 0.05
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=center)

        # Визуализация
        try:
            o3d.visualization.draw_geometries([pcd, axis])
        except Exception as vis_err:
            print(f"Визуализация недоступна: {vis_err}")

        return {
            "status": "ok",
            "first 5 points": first_5_points.tolist(),
            "input_file": filename,
            "points_total": points_count,
            "min_bound": min_bound.tolist(),
            "max_bound": max_bound.tolist(),
            "center": center.tolist(),
            "extent": extent.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/visualize", summary="Анализ и визуализация PLY-файла")
async def analyze_cloud(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        pcd = o3d.io.read_point_cloud(tmp_path)
        points_count = len(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(extent) * 0.1, origin=center)

        try:
            o3d.visualization.draw_geometries([pcd, axis])
        except Exception as e:
            print(f"Визуализация недоступна (ошибка с визуализатором на стороне сервера): {e}")

        return {
            "status": "ok",
            "message": f"Визуализировано {points_count} точек",
            "center": center.tolist(),
            "extent": extent.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа и визуализации на стороне сервера: {e}")