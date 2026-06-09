import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


METHOD_FILES: Dict[str, str] = {
    "Geo": "baseline_geometric.obj",
    "Mean": "baseline_mean.obj",
    "3DMM": "baseline_stage1.obj",
    "Sym": "baseline_symmetry.obj",
    "Ours": "2_reconstructed.obj",
}

GT_FILE = "0_ground_truth.obj"


def load_vertices(mesh_path: Path) -> np.ndarray:
    """Load mesh vertices from an OBJ/PLY file."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"File not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, process=False)

    # trimesh may return a Scene if the file contains multiple objects.
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices in file: {mesh_path}")

    return vertices


def nearest_neighbor_distances(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distances from src points to tgt points."""
    tree = cKDTree(tgt)
    distances, _ = tree.query(src, k=1, workers=-1)
    return distances


def chamfer_distance(pred: np.ndarray, gt: np.ndarray, scale: float = 1.0) -> float:
    """
    Symmetric Chamfer Distance.

    We use the average of bidirectional nearest-neighbor distances:
        CD = 0.5 * (mean d(pred -> gt) + mean d(gt -> pred))
    """
    d_pred_to_gt = nearest_neighbor_distances(pred, gt)
    d_gt_to_pred = nearest_neighbor_distances(gt, pred)
    cd = 0.5 * (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    return float(cd * scale)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray, scale: float = 1.0) -> float:
    """
    Symmetric Hausdorff Distance.

    We use the maximum of bidirectional nearest-neighbor distances:
        HD = max(max d(pred -> gt), max d(gt -> pred))
    """
    d_pred_to_gt = nearest_neighbor_distances(pred, gt)
    d_gt_to_pred = nearest_neighbor_distances(gt, pred)
    hd = max(np.max(d_pred_to_gt), np.max(d_gt_to_pred))
    return float(hd * scale)


def rmse_distance(pred: np.ndarray, gt: np.ndarray, scale: float = 1.0) -> float:
    """
    Point-wise RMSE.

    This metric assumes that pred and gt share the same vertex correspondence.
    """
    if pred.shape != gt.shape:
        return np.nan

    errors = np.linalg.norm(pred - gt, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(rmse * scale)


def compute_metrics_for_pair(pred_path: Path, gt_path: Path, scale: float) -> Dict[str, float]:
    """Compute CD, HD, and RMSE between one predicted mesh and ground truth."""
    pred_vertices = load_vertices(pred_path)
    gt_vertices = load_vertices(gt_path)

    return {
        "CD": chamfer_distance(pred_vertices, gt_vertices, scale=scale),
        "HD": hausdorff_distance(pred_vertices, gt_vertices, scale=scale),
        "RMSE": rmse_distance(pred_vertices, gt_vertices, scale=scale),
    }


def find_case_dirs(result_dir: Path) -> List[Path]:
    """Find result_mask_* folders."""
    case_dirs = sorted(
        [p for p in result_dir.iterdir() if p.is_dir() and p.name.startswith("result_mask_")],
        key=lambda x: x.name,
    )
    return case_dirs


def load_metadata(result_dir: Path, metadata_path: Optional[Path]) -> pd.DataFrame:
    """Load optional missing metadata CSV."""
    if metadata_path is None:
        candidate = result_dir / "missing_metadata.csv"
        if candidate.exists():
            metadata_path = candidate

    if metadata_path is None or not metadata_path.exists():
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_path)

    # Normalize possible folder/case column names.
    folder_col_candidates = ["folder", "case", "case_id", "result_folder", "sample"]
    folder_col = None
    for col in folder_col_candidates:
        if col in metadata.columns:
            folder_col = col
            break

    if folder_col is not None and folder_col != "folder":
        metadata = metadata.rename(columns={folder_col: "folder"})

    return metadata


def get_metadata_value(
    metadata: pd.DataFrame,
    folder: str,
    candidates: List[str],
    default_value=np.nan,
):
    """Fetch metadata value for a folder from possible column names."""
    if metadata.empty or "folder" not in metadata.columns:
        return default_value

    row = metadata[metadata["folder"].astype(str) == str(folder)]
    if row.empty:
        return default_value

    for col in candidates:
        if col in row.columns:
            return row.iloc[0][col]

    return default_value


def missing_ratio_group(ratio) -> str:
    """Group missing ratio into Low / Medium / High."""
    try:
        r = float(ratio)
    except Exception:
        return "Unknown"

    # Support both percentage values such as 35 and ratio values such as 0.35.
    if r > 1.0:
        r = r / 100.0

    if r < 0.30:
        return "Low (<30%)"
    if r <= 0.50:
        return "Medium (30%-50%)"
    return "High (>50%)"


def summarize_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation for each method."""
    rows = []
    for method in METHOD_FILES.keys():
        sub = df[df["Method"] == method]
        if sub.empty:
            continue

        rows.append(
            {
                "Method": method,
                "N": len(sub),
                "CD_mean": sub["CD"].mean(),
                "CD_std": sub["CD"].std(ddof=1),
                "HD_mean": sub["HD"].mean(),
                "HD_std": sub["HD"].std(ddof=1),
                "RMSE_mean": sub["RMSE"].mean(),
                "RMSE_std": sub["RMSE"].std(ddof=1),
            }
        )

    return pd.DataFrame(rows)


def summarize_ours_by_column(
    df: pd.DataFrame,
    column: str,
    output_column_name: str,
) -> pd.DataFrame:
    """Summarize Ours metrics grouped by missing ratio or missing pattern."""
    ours = df[df["Method"] == "Ours"].copy()
    if ours.empty or column not in ours.columns:
        return pd.DataFrame()

    rows = []
    for value, sub in ours.groupby(column, dropna=False):
        rows.append(
            {
                output_column_name: value,
                "N": len(sub),
                "CD_mean": sub["CD"].mean(),
                "CD_std": sub["CD"].std(ddof=1),
                "HD_mean": sub["HD"].mean(),
                "HD_std": sub["HD"].std(ddof=1),
                "RMSE_mean": sub["RMSE"].mean(),
                "RMSE_std": sub["RMSE"].std(ddof=1),
            }
        )

    return pd.DataFrame(rows)


def format_mean_std(mean_val: float, std_val: float, decimals: int = 4) -> str:
    """Format mean ± std for printing."""
    if pd.isna(mean_val):
        return "nan"
    if pd.isna(std_val):
        return f"{mean_val:.{decimals}f} ± nan"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print a compact mean ± std table."""
    if summary.empty:
        print("No summary available.")
        return

    print("\n=== Mean ± Std Summary ===")
    print(f"{'Method':<10} {'CD':<24} {'HD':<24} {'RMSE':<24}")
    print("-" * 86)

    for _, row in summary.iterrows():
        print(
            f"{row['Method']:<10} "
            f"{format_mean_std(row['CD_mean'], row['CD_std']):<24} "
            f"{format_mean_std(row['HD_mean'], row['HD_std']):<24} "
            f"{format_mean_std(row['RMSE_mean'], row['RMSE_std']):<24}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MCG-Face reconstruction metrics.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results_comparison_more40",
        help="Directory containing result_mask_* folders.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional path to missing_metadata.csv. If omitted, the script looks inside result_dir.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale factor applied to all distances. Use 100.0 for normalized coordinates if needed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save CSV outputs. Default: result_dir.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata) if args.metadata is not None else None
    metadata = load_metadata(result_dir, metadata_path)

    case_dirs = find_case_dirs(result_dir)
    if not case_dirs:
        raise RuntimeError(f"No result_mask_* folders found in {result_dir}")

    all_rows = []

    for case_dir in case_dirs:
        folder = case_dir.name
        gt_path = case_dir / GT_FILE

        if not gt_path.exists():
            print(f"[WARN] Missing ground truth for {folder}: {gt_path}")
            continue

        missing_ratio = get_metadata_value(
            metadata,
            folder,
            candidates=["missing_ratio", "ratio", "missing_rate", "removed_ratio"],
            default_value=np.nan,
        )
        missing_pattern = get_metadata_value(
            metadata,
            folder,
            candidates=["missing_pattern", "pattern", "mask_type", "defect_type"],
            default_value="Unknown",
        )
        ratio_group = missing_ratio_group(missing_ratio)

        for method, filename in METHOD_FILES.items():
            pred_path = case_dir / filename
            if not pred_path.exists():
                print(f"[WARN] Missing file for {folder} / {method}: {pred_path}")
                continue

            try:
                metrics = compute_metrics_for_pair(pred_path, gt_path, scale=args.scale)
            except Exception as exc:
                print(f"[ERROR] Failed on {folder} / {method}: {exc}")
                continue

            row = {
                "Folder": folder,
                "Method": method,
                "CD": metrics["CD"],
                "HD": metrics["HD"],
                "RMSE": metrics["RMSE"],
                "missing_ratio": missing_ratio,
                "missing_ratio_group": ratio_group,
                "missing_pattern": missing_pattern,
                "prediction_file": filename,
            }
            all_rows.append(row)

            print(
                f"[OK] {folder:<18} {method:<6} "
                f"CD={metrics['CD']:.4f}, HD={metrics['HD']:.4f}, RMSE={metrics['RMSE']:.4f}"
            )

    if not all_rows:
        raise RuntimeError("No metrics were computed. Please check input files.")

    metrics_df = pd.DataFrame(all_rows)

    per_sample_csv = output_dir / "metrics_per_sample.csv"
    mean_std_csv = output_dir / "metrics_mean_std.csv"
    ratio_csv = output_dir / "ours_missing_ratio_breakdown.csv"
    pattern_csv = output_dir / "ours_missing_pattern_breakdown.csv"

    metrics_df.to_csv(per_sample_csv, index=False)

    summary_df = summarize_mean_std(metrics_df)
    summary_df.to_csv(mean_std_csv, index=False)

    ratio_df = summarize_ours_by_column(
        metrics_df,
        column="missing_ratio_group",
        output_column_name="Missing Ratio",
    )
    if not ratio_df.empty:
        # Keep the expected order.
        order = ["Low (<30%)", "Medium (30%-50%)", "High (>50%)", "Unknown"]
        ratio_df["__order"] = ratio_df["Missing Ratio"].apply(
            lambda x: order.index(x) if x in order else len(order)
        )
        ratio_df = ratio_df.sort_values("__order").drop(columns=["__order"])
        ratio_df.to_csv(ratio_csv, index=False)

    pattern_df = summarize_ours_by_column(
        metrics_df,
        column="missing_pattern",
        output_column_name="Missing Pattern",
    )
    if not pattern_df.empty:
        pattern_df = pattern_df.sort_values("Missing Pattern")
        pattern_df.to_csv(pattern_csv, index=False)

    print_summary_table(summary_df)

    if not ratio_df.empty:
        print("\n=== Ours Missing-Ratio Breakdown ===")
        print(ratio_df.to_string(index=False))

    if not pattern_df.empty:
        print("\n=== Ours Missing-Pattern Breakdown ===")
        print(pattern_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  {per_sample_csv}")
    print(f"  {mean_std_csv}")
    if not ratio_df.empty:
        print(f"  {ratio_csv}")
    if not pattern_df.empty:
        print(f"  {pattern_csv}")


if __name__ == "__main__":
    main()
