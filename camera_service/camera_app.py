from fastapi import FastAPI
import camera  # твой camera.py
import numpy as np
import os

app = FastAPI()

# Папка для сохранений рядом с этим файлом
SAVE_DIR = os.path.join(os.path.dirname(__file__), "shared")
os.makedirs(SAVE_DIR, exist_ok=True)

# Камеру инициализируем лениво
cam = None

def get_camera():
    global cam
    if cam is None:
        cam = camera.init_camera()
    return cam


@app.get("/capture")
def capture():
    try:
        cam = get_camera()
        depth, amp = camera.capture_frame(cam, timeout_ms=500)
        points = camera.depth_to_pointcloud(depth, confidence=amp, conf_threshold=30.0)

        ply_file = os.path.join(SAVE_DIR, "last_pointcloud.ply")
        npy_file = os.path.join(SAVE_DIR, "last_depth.npy")

        camera.save_pointcloud_ply(ply_file, points)
        camera.save_depth_npy(npy_file, depth)

        return {
            "status": "ok",
            "ply_file": ply_file,
            "depth_file": npy_file,
            "points_count": int(points.shape[0])
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
