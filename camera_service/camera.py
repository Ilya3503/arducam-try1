import numpy as np
import ArducamDepthCamera as ac


def init_camera():
    cam = ac.ArducamCamera()
    if cam.open(ac.Connection.CSI, 0) != 0:
        raise RuntimeError("Failed to open camera")
    if cam.start(ac.FrameType.DEPTH) != 0:
        raise RuntimeError("Failed to start camera")
    return cam


def get_camera_info(cam):
    try:
        info = cam.getCameraInfo()
    except Exception:
        return {
            "width": 240,
            "height": 180,
            "fx": 200.0,
            "fy": 200.0,
            "cx": 240 / 2.0,
            "cy": 180 / 2.0
        }

    return {
        "width": getattr(info, "width", 240),
        "height": getattr(info, "height", 180),
        "fx": getattr(info, "fx", 200.0),
        "fy": getattr(info, "fy", 200.0),
        "cx": getattr(info, "cx", getattr(info, "width", 240) / 2.0),
        "cy": getattr(info, "cy", getattr(info, "height", 180) / 2.0),
    }



def capture_frame(cam, timeout_ms=2000):
    frame = cam.requestFrame(timeout_ms)
    if frame is None:
        raise RuntimeError("No frame from camera (timeout)")

    depth = frame.depth_data.copy()
    confidence = frame.confidence_data.copy()
    cam.releaseFrame(frame)

    if depth.shape != (180, 240):
        raise ValueError(f"Unexpected depth shape {depth.shape}, expected (180,240)")

    return depth, confidence


def depth_to_pointcloud(depth: np.ndarray,
                        intrinsics: dict,
                        confidence: np.ndarray = None,
                        conf_threshold: float = 30.0,
                        stride: int = 1):
    h, w = depth.shape

    fx = intrinsics.get("fx", 200.0)
    fy = intrinsics.get("fy", 200.0)
    cx = intrinsics.get("cx", w / 2.0)
    cy = intrinsics.get("cy", h / 2.0)

    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)

    Z = depth[::stride, ::stride]
    if confidence is not None:
        CONF = confidence[::stride, ::stride]
    else:
        CONF = None

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    mask = np.isfinite(pts[:, 2]) & (pts[:, 2] > 0.0)
    if CONF is not None:
        mask &= (CONF.reshape(-1) > conf_threshold)

    pts = pts[mask]
    return pts.astype(np.float32)



def save_pointcloud_ply(filename: str, points: np.ndarray):
    n = points.shape[0]
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_depth_npy(filename: str, depth: np.ndarray):
    np.save(filename, depth)
