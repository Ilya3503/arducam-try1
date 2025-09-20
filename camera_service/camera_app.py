from fastapi import FastAPI, HTTPException
import numpy as np
import os

import camera

app = FastAPI(
    title="Camera Service",
    description="Сервис по работе с камерой",
    openapi_tags=[
        {"name": "Служебные эндпоинты"},
        {"name": "Эндпоинты камеры", "description": "Команды по обработке изображений и CV"},
    ]
)


SHARED_DIR = os.environ.get("SHARED_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared"))
os.makedirs(SHARED_DIR, exist_ok=True)


cam = None


def get_camera():
    global cam
    if cam is None:
        cam = camera.init_camera()
    return cam


@app.get("/camera/health_check", tags=["Служебные эндпоинты"], summary="Проверка активности сервиса камеры")
async def health():
    return {"camera": "working!"}


@app.get("/camera/info", tags=["Эндпоинты камеры"], summary="Получение информации о камере")
async def get_camera_info_endpoint():
    try:
        cam = get_camera()
        cam_info = camera.get_camera_info(cam)
        return cam_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации о камере: {e}")


@app.get("/capture", tags=["Эндпоинты камеры"], summary="Получение и сохранение изображения")
def capture():
    try:
        cam = get_camera()
        depth, amp = camera.capture_frame(cam, timeout_ms=500)

        intrinsics = camera.get_camera_info(cam)
        points = camera.depth_to_pointcloud(depth, intrinsics, confidence=amp, conf_threshold=30.0)

        ply_file = os.path.join(SHARED_DIR, "last_pointcloud.ply")
        npy_file = os.path.join(SHARED_DIR, "last_depth.npy")

        camera.save_pointcloud_ply(ply_file, points)
        camera.save_depth_npy(npy_file, depth)

        return {
            "status": "ok",
            "ply_file": ply_file,
            "depth_file": npy_file,
            "points_count": int(points.shape[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения и сохранения данных от камеры: {e}")
