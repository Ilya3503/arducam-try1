from fastapi import FastAPI
from fastapi.responses import JSONResponse
import camera  # твой camera.py
import os


app = FastAPI()
cam = camera.init_camera()


SAVE_DIR = "/shared"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.get("/capture")
def capture():
    try:
        depth, amp = camera.capture_frame(cam, timeout_ms=500)
        points = camera.depth_to_pointcloud(depth, amplitude=amp, amp_threshold=30.0)

        ply_file = os.path.join(SAVE_DIR, "last_pointcloud.ply")
        npy_file = os.path.join(SAVE_DIR, "last_depth.npy")

        camera.save_pointcloud_ply(ply_file, points)
        camera.save_depth_npy(npy_file, depth)

        return JSONResponse(content={
            "status": "ok",
            "ply_file": ply_file,
            "depth_file": npy_file,
            "points_count": points.shape[0]
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
