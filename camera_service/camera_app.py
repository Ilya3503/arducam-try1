from fastapi import FastAPI
import camera
import os
from pathlib import Path

app = FastAPI(
    title="Camera Service",
    description="Сервис управления камерой",
    openapi_tags = [
        {"name": "Служебные эндпоинты"},
        {"name": "Эндпоинты камеры", "description": "Команды управления камерой"},
    ]
)


SHARED_DIR = Path(__file__).parent.parent / "shared"

cam = None

def get_camera():
    global cam
    if cam is None:
        cam = camera.init_camera()
    return cam

def get_camera_intrinsics(cam=None):

    return {
        "width": 240,
        "height": 180,
        "fx": 200.0,
        "fy": 200.0,
        "cx": 240 / 2.0,
        "cy": 180 / 2.0
    }


@app.get("/capture", tags=["Эндпоинты камеры"], summary="Захват изображения (создание depth map)")
def capture():
    try:
        cam = get_camera()
        depth, amp = camera.capture_frame(cam, timeout_ms=500)
        intrinsics = get_camera_intrinsics(cam)
        points = camera.depth_to_pointcloud(
            depth,
            intrinsics=intrinsics,
            confidence=amp,
            conf_threshold=30.0
        )

        ply_file = SHARED_DIR / "last_pointcloud.ply"
        npy_file = SHARED_DIR / "last_depth.npy"

        camera.save_pointcloud_ply(str(ply_file), points)
        camera.save_depth_npy(str(npy_file), depth)

        return {
            "status": "ok",
            "ply_file": ply_file,
            "depth_file": npy_file,
            "points_count": int(points.shape[0])
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}



@app.get("/camera/info", tags=["Эндпоинты камеры"], summary="Получение информации о камере: разрешение, fx/fy/cx/cy")
def get_camera_info_endpoint():
    try:
        cam = get_camera()
        cam_info = camera.get_camera_info(cam)
    except Exception as e:
        return {"status": "Ошибка получения информации о камере", "message": str(e)}

    return {"status": "Данные о камере получены успешно", "Camera Info": cam_info}