import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".ply":
        return o3d.io.read_point_cloud(str(path))

    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        pts = arr[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_point_cloud(pcd: o3d.geometry.PointCloud, file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".ply":
        o3d.io.write_point_cloud(str(path), pcd)
    elif path.suffix.lower() == ".npy":
        np.save(str(path), np.asarray(pcd.points))
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    return str(path)


# ---------------- Preprocessing ----------------
def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_noise(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


def remove_plane(pcd: o3d.geometry.PointCloud,
                 distance_threshold: float = 0.01,
                 ransac_n: int = 3,
                 num_iterations: int = 1000) -> o3d.geometry.PointCloud:
    _, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                   ransac_n=ransac_n,
                                   num_iterations=num_iterations)
    return pcd.select_by_index(inliers, invert=True)


def crop_roi(pcd: o3d.geometry.PointCloud,
             min_bound: Optional[tuple] = None,
             max_bound: Optional[tuple] = None) -> o3d.geometry.PointCloud:
    if min_bound is None or max_bound is None:
        return pcd
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)



def preprocess_point_cloud(input_path: str, output_path: str, voxel_size: float = 0.01) -> str:
    pcd = load_point_cloud(input_path)
    pcd = voxel_downsample(pcd, voxel_size=voxel_size)
    save_point_cloud(pcd, output_path)
    return output_path
