from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
from argparse import BooleanOptionalAction
import re
import fnmatch
import pandas as pd

from df_process_naive import filter_csv


BIN_RE = re.compile(
    r"^peaks_(?P<low>\d+(?:_\d)?)_(?P<high>\d+(?:_\d)?)_(?P<ts>\d{8}_\d{6}[+-]\d{4})\.csv$"
)

def parse_bin_key(p: Path) -> tuple[str, str, str] | None:
    """Return (low, high, ts) strings from filename like peaks_12_5_13_20251018_155817-0400.csv"""
    m = BIN_RE.match(p.name)
    if not m:
        return None
    return (m.group("low"), m.group("high"), m.group("ts"))

def latest_per_bin(files: list[Path]) -> list[Path]:
    """Keep the newest timestamped file per (low, high) bin."""
    best: dict[tuple[str, str], tuple[str, Path]] = {}
    for f in files:
        parsed = parse_bin_key(f)
        if not parsed:
            continue
        low, high, ts = parsed
        key = (low, high)
        prev = best.get(key)
        if prev is None or ts > prev[0]:  # lex compare works with YYYYmmdd_HHMMSS±ZZZZ
            best[key] = (ts, f)
    return [f for (_, f) in best.values()]

def gather_files(
    directory: Path,
    files: list[str] | None,
    includes: list[str] | None,
    excludes: list[str] | None,
    keep_latest: bool,
) -> list[Path]:
    # start with explicit files/globs or default peaks_*.csv
    candidates: set[Path] = set()
    if files:
        for pat in files:
            candidates.update(directory.glob(pat))
    else:
        candidates.update(directory.glob("peaks_*.csv"))

    # include filters (apply on basename)
    if includes:
        filtered = set()
        for f in candidates:
            if any(fnmatch.fnmatch(f.name, pat) for pat in includes):
                filtered.add(f)
        candidates = filtered

    # exclude filters (apply on basename)
    if excludes:
        candidates = {
            f for f in candidates if not any(fnmatch.fnmatch(f.name, pat) for pat in excludes)
        }

    # keep only files that look like peaks_* if we’re deduping per bin
    files_list = sorted(candidates)
    if keep_latest:
        files_list = latest_per_bin(files_list)

    return files_list

def run_one(file_path: Path, args, ts: str) -> pd.DataFrame:
    """Run filter_csv for a single input CSV and write a timestamped output CSV next to it."""
    out_csv_path = file_path.parent / f"{file_path.stem}_filtered_{ts}.csv"

    df = filter_csv(
        csv_path=file_path,
        out_csv_path=out_csv_path,     # mirror next to source
        band=args.band,
        asassn_csv=args.asassn_csv,
        vsx_csv=args.vsx_csv,
        min_dip_fraction=args.min_dip_fraction,
        min_cameras=args.min_cameras,
        max_power=args.max_power,
        min_period=args.min_period,
        max_period=args.max_period,
        min_time_span=200.0,       # defaults preserved even if filters are disabled
        min_points_per_day=0.05,
        min_sigma=3.0,
        match_radius_arcsec=args.match_radius_arcsec,
        n_helpers=args.n_helpers,
        apply_bns=args.apply_bns,
        apply_vsx_class=args.apply_vsx_class,
        apply_dip_dom=args.apply_dip_dom,
        apply_multi_camera=args.apply_multi_camera,
        apply_periodic=args.apply_periodic,
        apply_sparse=args.apply_sparse,
        apply_sigma=args.apply_sigma,
        chunk_size=args.chunk_size,
        tqdm_position_base=0,
    )

    print(f"[OK] {file_path} → {out_csv_path} ({len(df)} rows)")
    return df

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run filter_csv on one peaks CSV or on selected peaks_*.csv in a directory; outputs mirror source."
    )
    p.add_argument("csv_path", type=Path,
                   help="Input peaks CSV path OR a directory containing peaks_*.csv files.")
    # selection controls (only used when csv_path is a directory)
    p.add_argument("--files", nargs="+", default=None,
                   help="Specific file globs within the directory (e.g., 'peaks_12_5_13_*.csv').")
    p.add_argument("--include", action="append", default=None,
                   help="Additional basename globs to include; can be given multiple times.")
    p.add_argument("--exclude", action="append", default=None,
                   help="Basename globs to exclude; can be given multiple times.")
    p.add_argument("--latest-per-bin", action="store_true",
                   help="If multiple files exist per mag bin, keep only the latest (by timestamp in filename).")
    p.add_argument("--no-combined", action="store_true",
                   help="Do not write a combined CSV when processing a directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the files that would be processed and exit.")

    # pipeline knobs
    p.add_argument("--band", default="either", choices=["g", "v", "both", "either"])
    p.add_argument("--asassn-csv", default="results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv")
    p.add_argument("--vsx-csv", default="results_crossmatch/vsx_cleaned_20250926_1557.csv")
    p.add_argument("--min-dip-fraction", type=float, default=0.66)
    p.add_argument("--min-cameras", type=int, default=2)
    p.add_argument("--max-power", type=float, default=0.5)
    p.add_argument("--min-period", type=float, default=None)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--match-radius-arcsec", type=float, default=3.0)
    p.add_argument("--n-helpers", type=int, default=60)
    p.add_argument("--bns", dest="apply_bns", action=BooleanOptionalAction, default=True,
                   help="Apply the bright-nearby-star (catalog) join (use --no-bns to disable).")
    p.add_argument("--vsx-class", dest="apply_vsx_class", action=BooleanOptionalAction, default=True,
                   help="Append VSX variability classes (use --no-vsx-class to disable).")
    p.add_argument("--dip-dom", dest="apply_dip_dom", action=BooleanOptionalAction, default=False,
                   help="Apply the dip-dominated fraction filter (use --no-dip-dom to disable).")
    p.add_argument("--multi-camera", dest="apply_multi_camera", action=BooleanOptionalAction, default=False,
                   help="Apply the multi-camera filter (use --no-multi-camera to disable).")
    p.add_argument("--periodic", dest="apply_periodic", action=BooleanOptionalAction, default=False,
                   help="Apply the periodicity filter (use --no-periodic to disable).")
    p.add_argument("--sparse", dest="apply_sparse", action=BooleanOptionalAction, default=False,
                   help="Apply the sparse light-curve filter (use --no-sparse to disable).")
    p.add_argument("--sigma", dest="apply_sigma", action=BooleanOptionalAction, default=False,
                   help="Apply the sigma-based depth filter (use --no-sigma to disable).")
    p.add_argument("--chunk-size", type=int, default=None)

    args = p.parse_args()
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")

    in_path = args.csv_path

    # Case 1: directory — select subset, mirror outputs next to each input
    if in_path.is_dir():
        files = gather_files(
            directory=in_path,
            files=args.files,
            includes=args.include,
            excludes=args.exclude,
            keep_latest=args.latest_per_bin,
        )
        if not files:
            p.error(f"No matching files in {in_path}")

        if args.dry_run:
            print("Files that would be processed:")
            for f in files:
                print("  ", f)
            return 0

        dfs: list[pd.DataFrame] = []
        for f in files:
            dfs.append(run_one(f, args, ts))

        if not args.no_combined:
            combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            combined_csv = in_path / f"peaks_all_filtered_{ts}.csv"
            combined.to_csv(combined_csv, index=False)
            print(f"[COMBINED] {combined_csv} ({len(combined)} rows)")
        return 0

    # Case 2: single file (or a stem your module resolves) — mirror output next to it
    if in_path.is_file() or not in_path.exists():
        df = run_one(in_path, args, ts)
        print(f"Filtered rows: {len(df)}")
        return 0

    p.error(f"{in_path} is neither a file nor a directory")


if __name__ == "__main__":
    raise SystemExit(main())
