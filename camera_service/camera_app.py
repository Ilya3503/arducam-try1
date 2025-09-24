from fastapi import FastAPI
import camera  # твой camera.py
import numpy as np
import os

app = FastAPI(
    title="Camera Service",
    description="Сервис управления камерой",
    openapi_tags=[
        {"name": "Служебные эндпоинты"},
        {"name": "Эндпоинты камеры", "description": "Команды управления камерой"},
    ]
)

SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")
os.makedirs(SHARED_DIR, exist_ok=True)


# Камеру инициализируем лениво
cam = None

def get_camera():
    global cam
    if cam is None:
        cam = camera.init_camera()
    return cam

@app.get("/capture", tags=["Эндпоинты камеры"], summary="Захват изображения (создание depth map)")
def capture():
    try:
        cam = get_camera()
        depth, amp = camera.capture_frame(cam, timeout_ms=500)

        # Жёстко захардкоженные параметры камеры
        intrinsics = {
            "width": 240,
            "height": 180,
            "fx": 200.0,
            "fy": 200.0,
            "cx": 120.0,  # width / 2
            "cy": 90.0    # height / 2
        }

        points = camera.depth_to_pointcloud(
            depth,
            intrinsics=intrinsics,
            confidence=amp,
            conf_threshold=30.0
        )

        ply_file = os.path.join(SHARED_DIR, "last_pointcloud.ply")
        npy_file = os.path.join(SHARED_DIR, "last_depth.npy")

        camera.save_pointcloud_ply(str(ply_file), points)
        camera.save_depth_npy(str(npy_file), depth)

        return {
            "status": "ok",
            "ply_file": str(ply_file),
            "depth_file": str(npy_file),
            "points_count": int(points.shape[0])
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
