from pathlib import Path

import pandas as pd


def read_df_csv_naive(
    csv_path,
    require_both: bool = False,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
):
    """
    Read a CSV summarizing peak-finding results and return only rows where
    either band (default) or both bands (if ``require_both=True``) have
    a non-zero number of peaks. All columns are preserved.

    Behavior:
    - Coerces ``g_n_peaks`` and ``v_n_peaks`` to numeric; NaN -> 0.
    - Filters rows by (g_n_peaks > 0) OR (v_n_peaks > 0) by default.
      If ``require_both=True``, uses AND instead.
    - Adds a ``source_file`` column with the basename of ``csv_path`` for
      provenance.
    - Leaves array-like columns (e.g., ``g_peaks_idx``, ``g_peaks_jd``) as raw
      strings; no parsing is performed.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file to read.
    require_both : bool, optional
        If True, require non-zero peaks in both bands (AND). If False (default),
        accept non-zero peaks in either band (OR).
    out_csv_path : str or Path, optional
        Where to write the selected rows as CSV. If None and ``write_csv`` is
        True, writes next to the input as ``<stem>_selected_dippers.csv``.
    write_csv : bool, optional
        If True (default), write the filtered rows to CSV.
    index : bool, optional
        Whether to include the index when writing CSV (default False).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with all original columns retained, plus
        ``source_file``.
    """

    file = Path(csv_path)
    df = pd.read_csv(file).copy()

    # Ensure required columns exist
    for col in ("g_n_peaks", "v_n_peaks"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' is missing; cannot select nonzero-peak rows."
            )

    # Coerce counts to numeric and treat NaNs as zero
    df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
    df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

    # Selection mask: either band by default; both if requested
    if require_both:
        mask = (df["g_n_peaks"] > 0) & (df["v_n_peaks"] > 0)
    else:
        mask = (df["g_n_peaks"] > 0) | (df["v_n_peaks"] > 0)

    out = df.loc[mask].reset_index(drop=True)

    # Provenance
    out["source_file"] = file.name

    # Optionally write CSV of selected rows
    if write_csv:
        dest = (
            Path(out_csv_path)
            if out_csv_path is not None
            else file.parent / f"{file.stem}_selected_dippers.csv"
        )
        out.to_csv(dest, index=index)

    return out
