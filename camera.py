import numpy as np
import ArduCamDepthCamera as ac

def init_camera():
    cam = ac.ArducamCamera()
    if cam.init(ac.TOFConnect.CSI, ac.TOFOutput.DEPTH, 0) != 0:
        raise RuntimeError("Camera initialization failed")
    if cam.start() != 0:
        raise RuntimeError("Failed to start camera")
    return cam

def capture_frame(cam, timeout_ms=200):
    frame = cam.requestFrame(timeout_ms)
    if frame is None:
        raise RuntimeError("No frame from camera (timeout)")
    depth = frame.getDepthData().copy()
    amplitude = frame.getAmplitudeData().copy()
    cam.releaseFrame(frame)
    return depth, amplitude

def depth_to_pointcloud(depth: np.ndarray,
                        amplitude: np.ndarray = None,
                        amp_threshold: float = 30.0,
                        stride: int = 1):

    h, w = depth.shape

    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)
    Z = depth[::stride, ::stride]
    if amplitude is not None:
        AMP = amplitude[::stride, ::stride]
    else:
        AMP = None

    cx = w / 2.0
    cy = h / 2.0
    fx = fy = 200.0

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    mask = np.isfinite(pts[:, 2]) & (pts[:, 2] > 0.0)
    if AMP is not None:
        amp_flat = AMP.reshape(-1)
        mask &= (amp_flat > amp_threshold)

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
