import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List, Dict
import matplotlib.pyplot as plt


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

def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(pts[mask])
    return clean_pcd


def safe_crop(pcd: o3d.geometry.PointCloud,
              min_bound: Optional[tuple] = None,
              max_bound: Optional[tuple] = None) -> o3d.geometry.PointCloud:

    if min_bound is None or max_bound is None:
        return pcd
    pcd = clean_point_cloud(pcd)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)


def crop_points_numpy(pcd: o3d.geometry.PointCloud,
                      min_bound: Optional[tuple] = None,
                      max_bound: Optional[tuple] = None) -> o3d.geometry.PointCloud:

    if min_bound is None or max_bound is None:
        return pcd


    pcd = clean_point_cloud(pcd)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return o3d.geometry.PointCloud()

    mask = (
        (pts[:, 0] >= min_bound[0]) & (pts[:, 0] <= max_bound[0]) &
        (pts[:, 1] >= min_bound[1]) & (pts[:, 1] <= max_bound[1]) &
        (pts[:, 2] >= min_bound[2]) & (pts[:, 2] <= max_bound[2])
    )

    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])

    return cropped




def crop_roi(pcd: o3d.geometry.PointCloud,
             min_bound: Optional[tuple] = None,
             max_bound: Optional[tuple] = None) -> o3d.geometry.PointCloud:
    if min_bound is None or max_bound is None:
        return pcd
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)






def cluster_dbscan(pcd: o3d.geometry.PointCloud, eps: float, min_points: int = 20, max_points: int = None) -> List[o3d.geometry.PointCloud]:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError(f"Загружено пустое облако точек")

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        raise ValueError("В облаке не найдено ни одного кластера")

    unique_labels = np.unique(labels)
    clusters = []
    for lab in unique_labels:
        if lab == -1:
            continue  # -1 = шум
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue

        if max_points is not None and idx.size > max_points:
            continue

        cluster = pcd.select_by_index(idx.tolist())
        clusters.append(cluster)

    print(f"[cluster_dbscan] Найдено кластеров: {len(clusters)} (eps={eps}, min_points={min_points})")
    return clusters


def ensure_results_dir(results_dir: Path) -> Path:

    if results_dir.exists():
        clusters_dir = results_dir / "clusters"
        if clusters_dir.exists():
            return results_dir
        else:
            clusters_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        clusters_dir = results_dir / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        return results_dir


def save_cluster_files(clusters: List[o3d.geometry.PointCloud], clusters_dir: str) -> List[str]:
    clusters_dir = Path(clusters_dir)

    if not clusters_dir.exists():
        clusters_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for i, c in enumerate(clusters):
        fname = clusters_dir / f"cluster_{i:03d}.ply"
        try:
            o3d.io.write_point_cloud(str(fname), c)
            saved_paths.append(str(fname))
        except Exception as e:
            print(f"[save_cluster_files] Ошибка при сохранении {fname}: {e}")
    print(f"[save_cluster_files] Сохранено {len(saved_paths)} файлов в {clusters_dir}")
    return saved_paths


def remove_invalid_points_t(pcd: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
    pts = pcd.point.positions.numpy()
    mask = np.isfinite(pts).all(axis=1)
    clean_pcd = o3d.t.geometry.PointCloud()
    clean_pcd.point.positions = o3d.core.Tensor(pts[mask], dtype=pcd.point.positions.dtype)
    if "colors" in pcd.point:
        colors = pcd.point.colors.numpy()[mask]
        clean_pcd.point.colors = o3d.core.Tensor(colors, dtype=pcd.point.colors.dtype)
    return clean_pcd


def get_obb_for_cluster(cluster: o3d.t.geometry.PointCloud) -> dict:
    obb = cluster.get_oriented_bounding_box()

    # center и extent могут быть t.Tensor или numpy
    center = obb.center
    extent = obb.extent
    if hasattr(center, "numpy"):
        center = center.numpy()
    if hasattr(extent, "numpy"):
        extent = extent.numpy()
    center = center.tolist()
    extent = extent.tolist()

    # R точно t.Tensor
    R = obb.R
    if hasattr(R, "numpy"):
        R = R.numpy()
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))

    return {"center": center, "extent": extent, "yaw": yaw}



def create_and_save_annotated_pointcloud(pcd: o3d.geometry.PointCloud, clusters: List[o3d.geometry.PointCloud], results_dir: str) -> str:
    results_path = Path(results_dir)
    out_path = results_path / "annotated_pointcloud.ply"

    if not clusters:
        raise ValueError("[INFO]Ошибка: передан пустой список кластеров")

    # Цвета для кластеров (повторяются, если кластеров > 10)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]

    all_points, all_colors = [], []
    for i, cluster in enumerate(clusters):
        pts = np.asarray(cluster.points)
        if pts.size == 0:
            continue
        clr = np.tile(colors[i % len(colors)], (pts.shape[0], 1))
        all_points.append(pts)
        all_colors.append(clr)

    if not all_points:
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"[INFO] Saved plain PLY to {out_path}")
        return str(out_path)

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    o3d.io.write_point_cloud(str(out_path), merged)
    return str(out_path)



def save_position_json(result: dict, results_dir: str) -> str:
    import json
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "position.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[save_position_json] Сохранён JSON: {out}")
    return str(out)



def get_obb_geometries(clusters: list) -> list:
    obb_geoms = []
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]

    for i, cluster in enumerate(clusters):
        if len(cluster.points) == 0:
            continue

        obb = cluster.get_oriented_bounding_box()
        obb.color = colors[i % len(colors)]  # задаём цвет линии
        obb_geoms.append(obb)
    return obb_geoms


def create_annotated_pointcloud_with_obb(
    pcd: o3d.geometry.PointCloud,
    clusters: list,
    results_dir: str):

    results_path = Path(results_dir)
    out_path = results_path / "annotated_with_obb.ply"

    if not clusters:
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"[INFO] Saved plain PLY to {out_path}")
        return str(out_path)

    # Цвета кластеров
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]

    all_points, all_colors = [], []
    for i, cluster in enumerate(clusters):
        pts = np.asarray(cluster.points)
        if pts.size == 0:
            continue
        clr = np.tile(colors[i % len(colors)], (pts.shape[0], 1))
        all_points.append(pts)
        all_colors.append(clr)

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    # Добавляем obb рамки
    obb_geoms = get_obb_geometries(clusters)

    # Сохраняем отдельный файл с точками и рамками (если визуализатор поддерживает)
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"[INFO] Saved annotated + OBB PLY to {out_path}")

    # Можно также вернуть список obb для визуализатора
    return str(out_path), obb_geoms




def preprocess_point_cloud(
    input_path: str,
    output_path: str,
    voxel_size: float = 8.0,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    distance_threshold: float = 70.0,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_bound: tuple = (-231, -190, 474),
    max_bound: tuple = (264, 190, 670),
) -> str:
    pcd = load_point_cloud(input_path)
    pcd = voxel_downsample(pcd, voxel_size=voxel_size)
    pcd = remove_noise(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = remove_plane(pcd,  distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    pcd = crop_points_numpy(pcd, min_bound=min_bound, max_bound=max_bound)
    save_point_cloud(pcd, output_path)
    return output_path



def process_position(
    input_file: str,
    results_dir: str,
    eps: float = 30,
    min_points: int = 20,
    max_points: int = None,
    send_with_obb: bool = True
) -> dict:

    input_file = Path(input_file)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir = results_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    pcd = load_point_cloud(str(input_file))  # твоя функция загрузки
    pcd = remove_invalid_points_t(pcd)

    clusters = cluster_dbscan(pcd, eps=eps, min_points=min_points, max_points=max_points)
    clusters_info = []
    obbs_for_json = []

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]

    for i, c in enumerate(clusters):
        clusters_info.append({
            "id": i,
            "points_count": int(len(c.points))
        })

        if send_with_obb:
            info = get_obb_for_cluster(c)
            clusters_info[-1].update(info)
            col = colors[i % len(colors)].tolist()
            obb_entry = {
                "id": i,
                "center": info["center"],
                "extent": info["extent"],
                "yaw": info.get("yaw", 0.0),
                "color": col
            }
            obbs_for_json.append(obb_entry)

    clusters_paths = save_cluster_files(clusters, str(clusters_dir))
    annotated_path = create_and_save_annotated_pointcloud(pcd, clusters, str(results_dir))

    # Только если нужно OBB
    if send_with_obb:
        annotated_with_obb_path, obb_geoms = create_annotated_pointcloud_with_obb(pcd, clusters, str(results_dir))
    else:
        annotated_with_obb_path, obb_geoms = None, []

    result = {
        "status": "ok",
        "preprocessed_file": str(input_file),
        "results_dir": str(results_dir),
        "num_clusters": len(clusters),
        "clusters": clusters_info,
        "clusters_paths": clusters_paths,
        "annotated_ply": annotated_path,
        "obbs": obbs_for_json
    }

    save_position_json(result, str(results_dir))
    return result











