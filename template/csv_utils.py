"""
Simple CSV utilities for PS1 data organization.

Function-oriented approach for parsing skycell mapping and padding info.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data with basic error handling.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with CSV data
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"[CSV] Loaded CSV with {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"[CSV] Failed to load CSV {csv_path}: {e}")
        raise


def get_projections_from_csv(csv_path: str) -> list[str]:
    """Get unique projections from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        Sorted list of projection IDs
    """
    df = load_csv_data(csv_path)
    projections = df["projection"].astype(str).unique()
    projections = sorted(projections)

    logger.info(f"[CSV] Found {len(projections)} projections: {projections[:5]}...")
    return projections


def get_projection_rows(csv_path: str, projection: str) -> dict[int, list[str]]:
    """Get rows of skycells for a projection.

    Args:
        csv_path: Path to CSV file
        projection: PS1 projection ID

    Returns:
        Dictionary mapping row_id to list of skycells
    """
    df = load_csv_data(csv_path)
    proj_df = df[df["projection"].astype(str) == projection]

    rows = {}
    for _, row in proj_df.iterrows():
        row_id = int(row["row"])
        skycell = row["skycell"]

        if row_id not in rows:
            rows[row_id] = []
        rows[row_id].append(skycell)

    # Sort skycells within each row
    for row_id in rows:
        rows[row_id].sort()

    logger.info(f"[CSV] Found {len(rows)} rows for projection {projection}")
    return rows


def get_padding_info(csv_path: str, skycell: str) -> dict[str, list[str]]:
    """Get padding information for a skycell.

    Args:
        csv_path: Path to CSV file
        skycell: Skycell name

    Returns:
        Dictionary mapping direction to list of padding skycells
    """
    df = load_csv_data(csv_path)
    cell_row = df[df["NAME"] == skycell]

    if len(cell_row) == 0:
        logger.warning(f"[CSV] Skycell {skycell} not found in CSV")
        return {}

    row = cell_row.iloc[0]
    padding_info = {}

    # Check padding columns
    padding_columns = ["pad_skycell_top", "pad_skycell_right", "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_top_right", "pad_skycell_top_left", "pad_skycell_bottom_right", "pad_skycell_bottom_left"]

    for col in padding_columns:
        if col in row and pd.notna(row[col]) and row[col] != "":
            direction = col.replace("pad_skycell_", "")

            # Handle multiple skycells (could be comma-separated or slash-separated)
            cell_str = str(row[col]).replace("/", ",")
            padding_cells = cell_str.split(",")
            padding_cells = [cell.strip() for cell in padding_cells if cell.strip()]

            if padding_cells:
                padding_info[direction] = padding_cells

    logger.debug(f"[CSV] Found {len(padding_info)} padding directions for {skycell}")
    return padding_info


def get_all_padding_cells(csv_path: str, skycell_list: list[str]) -> dict[str, list[str]]:
    """Get all unique padding cells needed for a list of skycells.

    Args:
        csv_path: Path to CSV file
        skycell_list: List of skycell names

    Returns:
        Dictionary mapping skycell to list of needed padding cells
    """
    all_padding = {}
    unique_padding_cells = set()

    for skycell in skycell_list:
        padding_info = get_padding_info(csv_path, skycell)

        padding_cells = []
        for direction_cells in padding_info.values():
            padding_cells.extend(direction_cells)

        # Remove duplicates and current skycell
        padding_cells = list(set(padding_cells))
        if skycell in padding_cells:
            padding_cells.remove(skycell)

        all_padding[skycell] = padding_cells
        unique_padding_cells.update(padding_cells)

    logger.info(f"[CSV] Found {len(unique_padding_cells)} unique padding cells for {len(skycell_list)} skycells")
    return all_padding


def find_csv_file(data_root: str, sector: int, camera: int, ccd: int) -> str:
    """Find CSV file for given sector/camera/ccd.

    Args:
        data_root: Root data directory
        sector: TESS sector number
        camera: TESS camera number
        ccd: TESS CCD number

    Returns:
        Path to CSV file
    """
    import os

    # Use the correct CSV file pattern
    sector_str = f"sector_{sector:04d}"
    csv_path = f"{data_root}/skycell_pixel_mapping/{sector_str}/camera_{camera}/ccd_{ccd}/tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list.csv"

    if os.path.exists(csv_path):
        logger.info(f"[CSV] Found CSV file: {csv_path}")
        return csv_path

    # If main file not found, raise error with helpful message
    raise FileNotFoundError(f"Could not find CSV file: {csv_path}")
