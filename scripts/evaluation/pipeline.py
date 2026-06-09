import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


GT_FILE = "0_ground_truth.obj"
INPUT_FILE = "1_input_hole.obj"
OURS_FILE = "2_reconstructed.obj"

BASELINE_STAGE1_FILE = "baseline_stage1.obj"
BASELINE_SYMMETRY_FILE = "baseline_symmetry.obj"
BASELINE_MEAN_FILE = "baseline_mean.obj"
BASELINE_GEOMETRIC_FILE = "baseline_geometric.obj"

METADATA_FILE = "missing_metadata.csv"


def load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as Trimesh: {path}")

    return mesh


def save_mesh_like(vertices: np.ndarray, template_mesh: trimesh.Trimesh, out_path: Path) -> None:
    out_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=template_mesh.faces,
        process=False,
    )
    out_mesh.export(out_path)


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> bool:
    if not src.exists():
        return False

    if dst.exists() and not overwrite:
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_if_needed(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    if not src_dir.exists():
        return

    if dst_dir.exists() and not overwrite:
        return

    if dst_dir.exists() and overwrite:
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)


def collect_gt_meshes(case_dirs: List[Path]) -> List[Tuple[Path, trimesh.Trimesh]]:
    meshes = []

    for case_dir in case_dirs:
        gt_path = case_dir / GT_FILE
        if not gt_path.exists():
            continue

        try:
            mesh = load_mesh(gt_path)
            meshes.append((gt_path, mesh))
        except Exception as exc:
            print(f"[WARN] Failed to load GT mesh {gt_path}: {exc}")

    return meshes


def compute_mean_shape(gt_meshes: List[Tuple[Path, trimesh.Trimesh]]) -> Optional[Tuple[np.ndarray, trimesh.Trimesh]]:
    if not gt_meshes:
        return None

    template_mesh = gt_meshes[0][1]
    vertex_count = len(template_mesh.vertices)
    vertices_list = []

    for path, mesh in gt_meshes:
        if len(mesh.vertices) != vertex_count:
            print(f"[WARN] Skip mean-shape input with inconsistent vertex count: {path}")
            continue
        vertices_list.append(np.asarray(mesh.vertices, dtype=np.float64))

    if not vertices_list:
        return None

    mean_vertices = np.mean(np.stack(vertices_list, axis=0), axis=0)
    return mean_vertices, template_mesh


def generate_mean_baseline(
    case_dir: Path,
    mean_vertices: np.ndarray,
    template_mesh: trimesh.Trimesh,
    overwrite: bool = False,
) -> bool:
    out_path = case_dir / BASELINE_MEAN_FILE

    if out_path.exists() and not overwrite:
        return True

    try:
        save_mesh_like(mean_vertices, template_mesh, out_path)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to generate mean baseline for {case_dir.name}: {exc}")
        return False


def generate_geometric_baseline(case_dir: Path, overwrite: bool = False) -> bool:
    out_path = case_dir / BASELINE_GEOMETRIC_FILE

    if out_path.exists() and not overwrite:
        return True

    input_path = case_dir / INPUT_FILE
    if not input_path.exists():
        return False

    safe_copy(input_path, out_path, overwrite=overwrite)
    return True


def find_processed_meshes(processed_dir: Path) -> List[Path]:
    if not processed_dir.exists():
        print(f"[WARN] Processed directory not found: {processed_dir}")
        return []

    candidates = []
    for suffix in ("*.obj", "*.ply"):
        candidates.extend(sorted(processed_dir.glob(suffix)))

    return sorted(candidates)


def find_existing_case_dirs(source_result_dir: Optional[Path]) -> List[Path]:
    if source_result_dir is None or not source_result_dir.exists():
        return []

    return sorted(
        [p for p in source_result_dir.iterdir() if p.is_dir() and p.name.startswith("result_mask_")],
        key=lambda p: p.name,
    )


def create_case_folder_name(start_index: int, case_offset: int) -> str:
    return f"result_mask_{start_index + case_offset}"


def prepare_case_from_existing(
    existing_case_dir: Path,
    output_case_dir: Path,
    overwrite: bool = False,
) -> None:
    copy_tree_if_needed(existing_case_dir, output_case_dir, overwrite=overwrite)


def prepare_case_from_processed_mesh(
    mesh_path: Path,
    output_case_dir: Path,
    overwrite: bool = False,
) -> None:
    output_case_dir.mkdir(parents=True, exist_ok=True)

    gt_path = output_case_dir / GT_FILE
    if not gt_path.exists() or overwrite:
        safe_copy(mesh_path, gt_path, overwrite=True)


def ensure_required_files_report(case_dirs: List[Path]) -> None:
    required_files = [
        GT_FILE,
        INPUT_FILE,
        OURS_FILE,
        BASELINE_STAGE1_FILE,
        BASELINE_SYMMETRY_FILE,
        BASELINE_MEAN_FILE,
        BASELINE_GEOMETRIC_FILE,
    ]

    print("\n=== Required-file check ===")

    for case_dir in case_dirs:
        missing = [name for name in required_files if not (case_dir / name).exists()]
        if missing:
            print(f"[WARN] {case_dir.name}: missing {missing}")
        else:
            print(f"[OK]   {case_dir.name}: all required files found")


def read_existing_metadata(source_result_dir: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if source_result_dir is None:
        return {}

    metadata_path = source_result_dir / METADATA_FILE
    if not metadata_path.exists():
        return {}

    metadata = {}

    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row.get("folder") or row.get("case") or row.get("case_id")
            if folder:
                metadata[folder] = row

    return metadata


def write_metadata(
    output_dir: Path,
    case_dirs: List[Path],
    existing_metadata: Dict[str, Dict[str, str]],
) -> None:
    out_path = output_dir / METADATA_FILE

    fieldnames = [
        "folder",
        "missing_pattern",
        "missing_ratio",
        "source",
    ]

    rows = []

    for case_dir in case_dirs:
        folder = case_dir.name
        old = existing_metadata.get(folder, {})

        row = {
            "folder": folder,
            "missing_pattern": (
                old.get("missing_pattern")
                or old.get("pattern")
                or old.get("mask_type")
                or old.get("defect_type")
                or "Unknown"
            ),
            "missing_ratio": (
                old.get("missing_ratio")
                or old.get("ratio")
                or old.get("missing_rate")
                or old.get("removed_ratio")
                or ""
            ),
            "source": old.get("source") or "prepared_by_run_more40_full_pipeline",
        }
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Metadata saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare representative MCG-Face evaluation folders."
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed_masks",
        help="Directory containing processed complete meshes.",
    )
    parser.add_argument(
        "--source_result_dir",
        type=str,
        default=None,
        help="Optional existing result directory to copy from, e.g., results_comparison_more30.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_comparison_more40",
        help="Output directory for prepared evaluation folders.",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=40,
        help="Number of representative cases to prepare.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=801,
        help="Starting index for result_mask folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders or generated files.",
    )
    parser.add_argument(
        "--skip_mean_baseline",
        action="store_true",
        help="Skip generation of baseline_mean.obj.",
    )
    parser.add_argument(
        "--skip_geometric_baseline",
        action="store_true",
        help="Skip generation of baseline_geometric.obj fallback.",
    )

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    source_result_dir = Path(args.source_result_dir) if args.source_result_dir else None
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== MCG-Face representative evaluation preparation ===")
    print(f"Processed mesh directory : {processed_dir}")
    print(f"Source result directory  : {source_result_dir}")
    print(f"Output directory         : {output_dir}")
    print(f"Number of cases          : {args.num_cases}")
    print(f"Start index              : {args.start_index}")
    print(f"Overwrite                : {args.overwrite}")

    existing_case_dirs = find_existing_case_dirs(source_result_dir)
    processed_meshes = find_processed_meshes(processed_dir)

    prepared_case_dirs: List[Path] = []

    for i in range(args.num_cases):
        output_folder_name = create_case_folder_name(args.start_index, i)
        output_case_dir = output_dir / output_folder_name

        if i < len(existing_case_dirs):
            existing_case_dir = existing_case_dirs[i]
            print(f"[INFO] Copy existing case {existing_case_dir.name} -> {output_folder_name}")
            prepare_case_from_existing(
                existing_case_dir=existing_case_dir,
                output_case_dir=output_case_dir,
                overwrite=args.overwrite,
            )
        elif i < len(processed_meshes):
            mesh_path = processed_meshes[i]
            print(f"[INFO] Prepare GT from processed mesh {mesh_path.name} -> {output_folder_name}")
            prepare_case_from_processed_mesh(
                mesh_path=mesh_path,
                output_case_dir=output_case_dir,
                overwrite=args.overwrite,
            )
        else:
            print(f"[WARN] No source data available for case {output_folder_name}")
            output_case_dir.mkdir(parents=True, exist_ok=True)

        prepared_case_dirs.append(output_case_dir)

    if not args.skip_mean_baseline:
        gt_meshes = collect_gt_meshes(prepared_case_dirs)
        mean_result = compute_mean_shape(gt_meshes)

        if mean_result is None:
            print("[WARN] Could not compute mean shape. baseline_mean.obj will not be generated.")
        else:
            mean_vertices, template_mesh = mean_result
            generated_count = 0
            for case_dir in prepared_case_dirs:
                ok = generate_mean_baseline(
                    case_dir=case_dir,
                    mean_vertices=mean_vertices,
                    template_mesh=template_mesh,
                    overwrite=args.overwrite,
                )
                if ok:
                    generated_count += 1
            print(f"[OK] Mean baselines available/generated: {generated_count}")

    if not args.skip_geometric_baseline:
        generated_count = 0
        for case_dir in prepared_case_dirs:
            ok = generate_geometric_baseline(case_dir, overwrite=args.overwrite)
            if ok:
                generated_count += 1
        print(f"[OK] Geometric baselines available/generated: {generated_count}")

    existing_metadata = read_existing_metadata(source_result_dir)
    write_metadata(output_dir, prepared_case_dirs, existing_metadata)

    ensure_required_files_report(prepared_case_dirs)

    print("\nDone.")
    print("Next step:")
    print(
        "python scripts/evaluation/calc_metrics_mean_std.py "
        f"--result_dir {output_dir} --scale 100.0"
    )


if __name__ == "__main__":
    main()
