from pathlib import Path

import pandas as pd


def read_df_csv_naive(
    csv_path,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    band: str = "either",
):
    """
    Read peaks_[mag_bin].csv and return only rows where either band has a non-zero number of peaks. Optionally, output peaks_[mag_bin]_selected_dippers.csv. Optionally search for only g band, only v band, or both.
    """

    file = Path(csv_path)
    df = pd.read_csv(file).copy()

    for col in ("g_n_peaks", "v_n_peaks"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' is missing; cannot select nonzero-peak rows."
            )

    df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
    df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

    band_key = band.lower()
    if band_key not in {"g", "v", "both", "either"}:
        raise ValueError("band must be one of 'g', 'v', 'both', 'either'")

    if band_key == "g":
        mask = df["g_n_peaks"] > 0
    elif band_key == "v":
        mask = df["v_n_peaks"] > 0
    elif band_key == "both":
        mask = (df["g_n_peaks"] > 0) & (df["v_n_peaks"] > 0)
    else:
        mask = (df["g_n_peaks"] > 0) | (df["v_n_peaks"] > 0)

    out = df.loc[mask].reset_index(drop=True)
    out["source_file"] = file.name

    if write_csv:
        dest = (
            Path(out_csv_path)
            if out_csv_path is not None
            else file.parent / f"{file.stem}_selected_dippers.csv"
        )
        out.to_csv(dest, index=index)

    return out


# implement Brayden's camera filter 1, camera filter 2?

# stats you have to work with


#
    #   mag_bin
    #   asas_sn_id
    #   index_num
    #   index_csv
    #   lc_dir
    #   dat_path
    #   raw_path
    #   g_n_peaks
    #   g_mean_mag
    #   g_peaks_idx
    #   g_peaks_jd
    #   v_n_peaks
    #   v_mean_mag
    #   v_peaks_idx
    #   v_peaks_jd
    #   jd_first
    #   jd_last
    #   n_rows_g
    #   n_rows_v
    
# in lc_dips.naive_dip_finder
    #    n_dip_runs,
    #    n_jump_runs,
    #    n_dip_points,
    #    n_jump_points,
    #    most_recent_dip,
    #    most_recent_jump,
    #    max_depth,
    #    max_height,
    #    max_dip_duration,
    #    max_jump_duration,
    #    dip_fraction
    #    jump_fraction

# in the raw file (which can instead be appended to the output df, but you need to make changes in lc_metrics probably by making lc_metrics ingest corresponding raw with lc_utils.read_lc_raw() )

    #   camera#
    #   median
    #   sig1_low
    #   sig1_high
    #   p90_low
    #   p90_high



def is_dip_dominated(metrics_dict, min_dip_fraction=0.66):
    """
    returns True if the the dip fraction from the metrics dict is above a certain value, currently 2/3
    """
    if ~np.isnan(metrics_dict["dip_fraction"]) and metrics_dict["dip_fraction"] >= min_dip_fraction:
            return True
    else :
        return False
 

def multi_camera_confirmation():
    pass



def filter_df(
    df,
    *,
    min_rows_g=None,
    min_rows_v=None,
    min_g_n_peaks=None,
    min_v_n_peaks=None,
    min_g_dip_fraction=None,
    max_g_jump_fraction=None,
    min_v_dip_fraction=None,
    max_v_jump_fraction=None,
    min_g_n_dip_runs=None,
    min_v_n_dip_runs=None,
    min_g_max_depth=None,
    min_v_max_depth=None,
    require_g_dip_dominated=None,
    require_v_dip_dominated=None,
):
    """
    Return a filtered copy of a dip_finder DataFrame.

    Pass whichever thresholds you want to explore; omitted parameters are not
    enforced. ``require_*_dip_dominated`` accepts ``True``/``False`` to demand a
    specific dip-dominated classification from :func:`lc_metrics.is_dip_dominated`.
    """

    mask = pd.Series(True, index=df.index)

    def _col(name):
        if name not in df.columns:
            raise KeyError(
                f"Column '{name}' is missing; regenerate parquet with dip_finder to include it."
            )
        return df[name]

    def _gte(name, value):
        nonlocal mask
        if value is None:
            return
        mask &= _col(name).fillna(-np.inf) >= value

    def _lte(name, value):
        nonlocal mask
        if value is None:
            return
        mask &= _col(name).fillna(np.inf) <= value

    def _require_bool(name, desired):
        nonlocal mask
        if desired is None:
            return
        series = _col(name).fillna(False).astype(bool)
        mask &= (series == bool(desired))

    # point-count constraints
    _gte("n_rows_g", min_rows_g)
    _gte("n_rows_v", min_rows_v)

    # peak counts
    _gte("g_n_peaks", min_g_n_peaks)
    _gte("v_n_peaks", min_v_n_peaks)

    # dip/jump fractions
    _gte("g_dip_fraction", min_g_dip_fraction)
    _lte("g_jump_fraction", max_g_jump_fraction)
    _gte("v_dip_fraction", min_v_dip_fraction)
    _lte("v_jump_fraction", max_v_jump_fraction)

    # run counts
    _gte("g_n_dip_runs", min_g_n_dip_runs)
    _gte("v_n_dip_runs", min_v_n_dip_runs)

    # depth thresholds
    _gte("g_max_depth", min_g_max_depth)
    _gte("v_max_depth", min_v_max_depth)

    # dip-dominated requirement
    _require_bool("g_is_dip_dominated", require_g_dip_dominated)
    _require_bool("v_is_dip_dominated", require_v_dip_dominated)

    return df.loc[mask].reset_index(drop=True)
