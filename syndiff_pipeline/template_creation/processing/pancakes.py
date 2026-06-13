"""
PanCAKES v2.0 - Advanced Astronomical Image Processing Pipeline

This module provides a modern, high-performance implementation for processing
TESS (Transiting Exoplanet Survey Satellite) Full Frame Images and matching them
with PanSTARRS1 (PS1) SkyCell data using advanced computational techniques.

Author: Generated from optimization notebook analysis
Version: 2.0
"""

# Standard library imports
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import WCS, FITSFixedWarning
from mocpy import MOC
from numba import jit
from shapely.geometry import Polygon
from tqdm import tqdm

from syndiff_pipeline.template_creation.processing.downsample import load_cluster_template_job_payload

# Suppress common warnings
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# PADDING AND SKYCELL UTILITIES
# ============================================================================


@dataclass
class PaddingRequirements:
    """Flags indicating which sides/corners require padding based on TESS mapping."""

    top: bool = False
    bottom: bool = False
    left: bool = False
    right: bool = False
    top_left: bool = False
    top_right: bool = False
    bottom_left: bool = False
    bottom_right: bool = False

    def any_needed(self) -> bool:
        """Check if any side needs padding."""
        return any([self.top, self.bottom, self.left, self.right, self.top_left, self.top_right, self.bottom_left, self.bottom_right])

    def good_side_fail(self):
        """Check if padding is needed for sides that should never need it."""
        return any([self.bottom, self.left, self.top_left, self.bottom_left, self.bottom_right])

    def to_list(self):
        """Convert to list of boolean values."""
        return [self.top, self.right, self.top_right, self.bottom, self.left, self.bottom_left, self.bottom_right, self.top_left]

    def from_list(self, padding_list):
        """Set values from a list of booleans."""
        if len(padding_list) != 8:
            raise ValueError("Padding list must have exactly 8 elements")

        self.top = padding_list[0]
        self.right = padding_list[1]
        self.top_right = padding_list[2]
        self.bottom = padding_list[3]
        self.left = padding_list[4]
        self.bottom_left = padding_list[5]
        self.bottom_right = padding_list[6]
        self.top_left = padding_list[7]

        return self


def get_projection_cell_id(skycell_name):
    """
    Parse skycell name to extract projection, cell, y, and x coordinates.

    Args:
        skycell_name (str): Skycell name in format 'skycell.projection.cell'

    Returns:
        tuple: (projection, cell, y, x)
    """
    _, projection, cell = skycell_name.split(".")
    if _ != "skycell":
        print("Invalid skycell name format")
    projection = int(projection)
    y = int(cell[1])
    x = int(cell[2])
    return projection, cell, y, x


def check_tess_mapping_padding(mapping_data: np.ndarray, pad_distance: int = 500, edge_exclusion: int = 10) -> PaddingRequirements:
    """
    Analyze a TESS mapping FITS to determine which borders require padding.

    Expects that pixels mapped to this skycell are non-zero; edges with any non-zero within
    the first/last pad_distance (excluding a small inner edge) are flagged for padding.

    Args:
        mapping_data (np.ndarray): 2D mapping array
        pad_distance (int): Distance from edge to check for padding requirements
        edge_exclusion (int): Pixels to exclude from very edge

    Returns:
        PaddingRequirements: Object indicating which sides need padding
    """
    padding = PaddingRequirements()
    # Top
    padding.top = bool(np.any(mapping_data[-(pad_distance + edge_exclusion) : -edge_exclusion, edge_exclusion:-edge_exclusion] != -1))
    # Bottom
    padding.bottom = bool(np.any(mapping_data[edge_exclusion : pad_distance + edge_exclusion, edge_exclusion:-edge_exclusion] != -1))
    # Left
    padding.left = bool(np.any(mapping_data[edge_exclusion:-edge_exclusion, edge_exclusion : pad_distance + edge_exclusion] != -1))
    # Right
    padding.right = bool(np.any(mapping_data[edge_exclusion:-edge_exclusion, -(pad_distance + edge_exclusion) : -edge_exclusion] != -1))
    # Corners
    padding.top_left = bool(np.any(mapping_data[-(pad_distance + edge_exclusion) : -edge_exclusion, edge_exclusion : pad_distance + edge_exclusion] != -1))
    padding.top_right = bool(np.any(mapping_data[-(pad_distance + edge_exclusion) : -edge_exclusion, -(pad_distance + edge_exclusion) : -edge_exclusion] != -1))
    padding.bottom_left = bool(np.any(mapping_data[edge_exclusion : pad_distance + edge_exclusion, edge_exclusion : pad_distance + edge_exclusion] != -1))
    padding.bottom_right = bool(np.any(mapping_data[edge_exclusion : pad_distance + edge_exclusion, -(pad_distance + edge_exclusion) : -edge_exclusion] != -1))
    return padding


def get_padding_corners(skycell_row, ps1_wcs, padding_side, pad_size=500, edge_exclusion=10):
    """
    Get the corners of a padding region for a given skycell and padding side.

    Args:
        skycell_row: Row from the skycell DataFrame
        padding_side: One of 'top', 'right', 'top_right', etc.
        ps1_wcs: Optional WCS object (to avoid recreation)
        pad_size: Size of the padding in pixels
        edge_exclusion: Overlap with the original skycell in pixels

    Returns:
        list: List of [RA, DEC] corner coordinates for the padding region
    """
    # Get dimensions needed for padding calculations
    naxis1 = skycell_row["NAXIS1"]
    naxis2 = skycell_row["NAXIS2"]

    # Define coordinates for each padding region
    # Each region extends pad_size pixels outside the edge and edge_exclusion pixels inside
    if padding_side == "top":
        x_coords = [0, naxis1, naxis1, 0]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
    elif padding_side == "right":
        x_coords = [naxis1 - edge_exclusion, naxis1 + pad_size, naxis1 + pad_size, naxis1 - edge_exclusion]
        y_coords = [0, 0, naxis2, naxis2]
    elif padding_side == "bottom":
        x_coords = [0, naxis1, naxis1, 0]
        y_coords = [-pad_size, -pad_size, edge_exclusion, edge_exclusion]
    elif padding_side == "left":
        x_coords = [-pad_size, edge_exclusion, edge_exclusion, -pad_size]
        y_coords = [0, 0, naxis2, naxis2]
    elif padding_side == "top_right":
        x_coords = [naxis1 - edge_exclusion, naxis1 + pad_size, naxis1 + pad_size, naxis1 - edge_exclusion]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
    elif padding_side == "bottom_right":
        x_coords = [naxis1 - edge_exclusion, naxis1 + pad_size, naxis1 + pad_size, naxis1 - edge_exclusion]
        y_coords = [-pad_size, -pad_size, edge_exclusion, edge_exclusion]
    elif padding_side == "bottom_left":
        x_coords = [-pad_size, edge_exclusion, edge_exclusion, -pad_size]
        y_coords = [-pad_size, -pad_size, edge_exclusion, edge_exclusion]
    elif padding_side == "top_left":
        x_coords = [edge_exclusion, -pad_size, -pad_size, edge_exclusion]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
        x_coords = [-pad_size, edge_exclusion, edge_exclusion, -pad_size]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
    else:
        raise ValueError(f"Unknown padding side: {padding_side}")

    # Convert pixel coordinates to world coordinates (RA, DEC)
    world_coords = ps1_wcs.wcs_pix2world(np.vstack([x_coords, y_coords]).T, 0)

    # Return corners in [RA, DEC] format
    return [[ra, dec] for ra, dec in world_coords]


def get_padding_center(corners):
    """Calculate the center point of a padding region."""
    ra_avg = np.mean([corner[0] for corner in corners])
    dec_avg = np.mean([corner[1] for corner in corners])
    return ra_avg, dec_avg


def calculate_distance(ra1, dec1, ra2, dec2):
    """Calculate angular distance between two points in degrees."""
    c1 = SkyCoord(ra1 * u.degree, dec1 * u.degree, frame="icrs")
    c2 = SkyCoord(ra2 * u.degree, dec2 * u.degree, frame="icrs")
    return c1.separation(c2).degree


def calculate_overlap(region1, region2):
    """Calculate the percentage of region1 covered by region2."""
    try:
        intersection = region1.intersection(region2)
        if intersection.is_empty:
            return 0.0
        return (intersection.area / region1.area) * 100.0
    except Exception as e:
        print(f"Error calculating overlap: {e}")
        return 0.0


def find_best_padding_skycell(target_skycell, padding_corners, all_skycells):
    """
    Find the best skycell for padding based on coverage and proximity.

    Args:
        target_skycell: The skycell that needs padding
        padding_corners: The corners of the padding region
        all_skycells: DataFrame of all available skycells

    Returns:
        dict: Results containing best skycell(s) info and coverage analysis
    """
    # Create padding region polygon
    padding_region = Polygon(padding_corners)
    padding_center_ra, padding_center_dec = get_padding_center(padding_corners)

    # PS1 skycell width in degrees (approximate)
    ps1_width = 0.4  # degrees

    # Calculate search radius (diagonal of skycell * sqrt(2))
    search_radius = (ps1_width / 2) * np.sqrt(2)

    # Find all potentially overlapping skycells
    overlapping_candidates = []

    for _, candidate in all_skycells.iterrows():
        # Skip the target skycell itself
        if candidate["NAME"] == target_skycell["NAME"]:
            continue

        # Calculate center-to-center distance
        candidate_center_ra = np.mean([candidate[f"RA_Corner{i}"] for i in range(1, 5)])
        candidate_center_dec = np.mean([candidate[f"DEC_Corner{i}"] for i in range(1, 5)])
        distance = calculate_distance(padding_center_ra, padding_center_dec, candidate_center_ra, candidate_center_dec)

        # Filter by distance to reduce computation
        if distance < search_radius:
            # Create candidate polygon
            candidate_corners = [[candidate[f"RA_Corner{i}"], candidate[f"DEC_Corner{i}"]] for i in range(1, 5)]
            candidate_polygon = Polygon(candidate_corners)

            # Calculate overlap
            coverage = calculate_overlap(padding_region, candidate_polygon)

            if coverage > 0:
                overlapping_candidates.append({"skycell_id": candidate["NAME"], "projection": candidate["projection"], "coverage": coverage, "distance": distance, "polygon": candidate_polygon})

    # Sort by coverage (primary) and distance (secondary)
    overlapping_candidates.sort(key=lambda x: (-x["coverage"], x["distance"]))

    if not overlapping_candidates:
        return {"status": "no_overlap", "best_match": None, "coverage": 0, "combined_solutions": []}

    # Check if we have 100% coverage with the best candidate
    if overlapping_candidates[0]["coverage"] >= 99.9:  # Allow for small numerical errors
        return {"status": "full_coverage", "best_match": overlapping_candidates[0]["skycell_id"], "best_match_proj": overlapping_candidates[0]["projection"], "coverage": overlapping_candidates[0]["coverage"], "distance": overlapping_candidates[0]["distance"], "combined_solutions": []}

    # Try to find combinations that provide full coverage
    combined_solutions = []

    # Try all pairs of candidates
    for i in range(len(overlapping_candidates)):
        for j in range(i + 1, len(overlapping_candidates)):
            combined_polygon = overlapping_candidates[i]["polygon"].union(overlapping_candidates[j]["polygon"])
            combined_coverage = calculate_overlap(padding_region, combined_polygon)

            if combined_coverage >= 99.9:  # Allow for small numerical errors
                combined_solutions.append(
                    {
                        "skycells": [overlapping_candidates[i]["skycell_id"], overlapping_candidates[j]["skycell_id"]],
                        "projections": [overlapping_candidates[i]["projection"], overlapping_candidates[j]["projection"]],
                        "coverage": combined_coverage,
                        "avg_distance": (overlapping_candidates[i]["distance"] + overlapping_candidates[j]["distance"]) / 2,
                    }
                )

    # Sort combined solutions by average distance
    combined_solutions.sort(key=lambda x: x["avg_distance"])

    return {
        "status": "partial_coverage" if not combined_solutions else "combined_coverage",
        "best_match": overlapping_candidates[0]["skycell_id"],
        "best_match_proj": overlapping_candidates[0]["projection"],
        "coverage": overlapping_candidates[0]["coverage"],
        "distance": overlapping_candidates[0]["distance"],
        "combined_solutions": combined_solutions,
    }


def parse_special_padding_flags(flags_str):
    """Parse the special_padding_flags string into a list of booleans."""
    if not isinstance(flags_str, str):
        return [False] * 8

    try:
        flags = eval(flags_str)
        if isinstance(flags, list) and len(flags) == 8:
            return flags
        return [False] * 8
    except Exception:
        return [False] * 8


# ============================================================================
# NUMBA-ACCELERATED COORDINATE TRANSFORMATION FUNCTIONS
# ============================================================================


@jit(nopython=True)
def inverse_tan_projection(xi, eta, crval1, crval2):
    """
    Perform inverse tangential projection using Numba acceleration.

    Args:
        xi (float): Projected x coordinate in degrees
        eta (float): Projected y coordinate in degrees
        crval1 (float): Reference RA in degrees
        crval2 (float): Reference Dec in degrees

    Returns:
        tuple: (RA, Dec) in degrees
    """
    xi_rad = np.deg2rad(xi)
    eta_rad = np.deg2rad(eta)
    ra0 = np.deg2rad(crval1)
    dec0 = np.deg2rad(crval2)

    if np.allclose(xi, 0) and np.allclose(eta, 0):
        return (crval1, crval2)

    rho = np.sqrt(xi_rad**2 + eta_rad**2)
    c = np.arctan(rho)

    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    dec = np.arcsin(cos_c * sin_dec0 + (eta_rad * sin_c * cos_dec0) / rho)

    y_term = xi_rad * sin_c
    x_term = rho * cos_dec0 * cos_c - eta_rad * sin_dec0 * sin_c
    ra = ra0 + np.arctan2(y_term, x_term)

    return (np.rad2deg(ra), np.rad2deg(dec))


@jit(nopython=True)
def calculate_radec(x, y, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    """
    Calculate RA/Dec from pixel coordinates using WCS parameters.

    Args:
        x, y (float): Pixel coordinates
        crval1, crval2 (float): Reference world coordinates
        crpix1, crpix2 (float): Reference pixel coordinates
        pc1_1, pc1_2, pc2_1, pc2_2 (float): PC matrix elements
        cdelt1, cdelt2 (float): Coordinate deltas

    Returns:
        tuple: (RA, Dec) in degrees
    """
    u = (x - crpix1 + 1) * cdelt1
    v = (y - crpix2 + 1) * cdelt2

    xi = pc1_1 * u + pc1_2 * v
    eta = pc2_1 * u + pc2_2 * v

    ra, dec = inverse_tan_projection(xi, eta, crval1, crval2)
    return ra, dec


@jit(nopython=True)
def calculate_radec_corners_numba(buffer, naxis1, naxis2, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    """
    Calculate RA/Dec for corners of multiple images with buffer.

    Args:
        buffer (float): Buffer size in pixels
        naxis1, naxis2 (array): Image dimensions
        WCS parameters: Arrays of WCS transformation parameters

    Returns:
        ndarray: Shape (N, 4, 2) array of corner coordinates
    """
    if not (naxis1.shape == naxis2.shape == crval1.shape == crval2.shape == crpix1.shape == crpix2.shape == pc1_1.shape == pc1_2.shape == pc2_1.shape == pc2_2.shape == cdelt1.shape == cdelt2.shape):
        raise ValueError("All input arrays must have the same shape")

    ra_dec = np.empty((crval1.shape[0], 4, 2), dtype=np.float64)
    for i in range(crval1.shape[0]):
        x = np.array([buffer, buffer, naxis1[i] - buffer, naxis1[i] - buffer])
        y = np.array([buffer, naxis2[i] - buffer, naxis2[i] - buffer, buffer])
        for c in range(4):
            ra_dec[i, c, 0], ra_dec[i, c, 1] = calculate_radec(x[c], y[c], crval1[i], crval2[i], crpix1[i], crpix2[i], pc1_1[i], pc1_2[i], pc2_1[i], pc2_2[i], cdelt1[i], cdelt2[i])
    return ra_dec


@jit(nopython=True)
def calculate_radec_corners_shift_numba(buffer_large, buffer_small, buffer_normal, cell_x, cell_y, naxis1, naxis2, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    # Check array shapes
    if not (naxis1.shape == naxis2.shape == crval1.shape == crval2.shape == crpix1.shape == crpix2.shape == pc1_1.shape == pc1_2.shape == pc2_1.shape == pc2_2.shape == cdelt1.shape == cdelt2.shape):
        raise ValueError("All input arrays must have the same shape")

    ra_dec = np.empty((crval1.shape[0], 4, 2), dtype=np.float64)
    for i in range(crval1.shape[0]):
        if cell_x[i] == 0:
            x = np.array([buffer_normal, buffer_normal, naxis1[i] - buffer_small, naxis1[i] - buffer_small])
        elif cell_x[i] == 9:
            x = np.array([buffer_large, buffer_large, naxis1[i] - buffer_normal, naxis1[i] - buffer_normal])
        else:
            x = np.array([buffer_large, buffer_large, naxis1[i] - buffer_small, naxis1[i] - buffer_small])

        if cell_y[i] == 0:
            y = np.array([buffer_normal, naxis2[i] - buffer_small, naxis2[i] - buffer_small, buffer_normal])
        elif cell_y[i] == 9:
            y = np.array([buffer_large, naxis2[i] - buffer_normal, naxis2[i] - buffer_normal, buffer_large])
        else:
            y = np.array([buffer_large, naxis2[i] - buffer_small, naxis2[i] - buffer_small, buffer_large])

        for c in range(4):
            ra_dec[i, c, 0], ra_dec[i, c, 1] = calculate_radec(x[c], y[c], crval1[i], crval2[i], crpix1[i], crpix2[i], pc1_1[i], pc1_2[i], pc2_1[i], pc2_2[i], cdelt1[i], cdelt2[i])

    return ra_dec


@jit(nopython=True)
def calculate_radec_center_numba(naxis1, naxis2, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    """
    Calculate RA/Dec for centers of multiple images.

    Args:
        naxis1, naxis2 (array): Image dimensions
        WCS parameters: Arrays of WCS transformation parameters

    Returns:
        ndarray: Shape (N, 2) array of center coordinates
    """
    if not (naxis1.shape == naxis2.shape == crval1.shape == crval2.shape == crpix1.shape == crpix2.shape == pc1_1.shape == pc1_2.shape == pc2_1.shape == pc2_2.shape == cdelt1.shape == cdelt2.shape):
        raise ValueError("All input arrays must have the same shape")

    ra_dec = np.empty((crval1.shape[0], 2), dtype=np.float64)
    for i in range(crval1.shape[0]):
        ra_dec[i, 0], ra_dec[i, 1] = calculate_radec(naxis1[i] // 2, naxis2[i] // 2, crval1[i], crval2[i], crpix1[i], crpix2[i], pc1_1[i], pc1_2[i], pc2_1[i], pc2_2[i], cdelt1[i], cdelt2[i])
    return ra_dec


@jit(nopython=True)
def create_closest_center_array_numba(rust_result_flat, rust_result_lengths, projection_centers_x, projection_centers_y, pixel_coords, total_size):
    """
    Create array mapping each pixel to its closest skycell center.

    Args:
        rust_result_flat (array): Flattened pixel IDs from MOC filtering
        rust_result_lengths (array): Length of each skycell's pixel list
        projection_centers_x, projection_centers_y (array): Skycell projection centers
        pixel_coords (array): Pixel coordinates [y, x] for each pixel
        total_size (int): Total number of pixels

    Returns:
        ndarray: Array mapping pixel IDs to skycell indices
    """
    output_array = np.full(total_size, -1, dtype=np.int32)
    min_distances = np.full(total_size, np.inf, dtype=np.float64)

    start_idx = 0
    for list_idx in range(len(rust_result_lengths)):
        end_idx = start_idx + rust_result_lengths[list_idx]

        center_x = projection_centers_x[list_idx]
        center_y = projection_centers_y[list_idx]

        for i in range(start_idx, end_idx):
            pixel_id = rust_result_flat[i]
            pixel_x = pixel_coords[pixel_id, 1]  # x coordinate
            pixel_y = pixel_coords[pixel_id, 0]  # y coordinate

            # Calculate distance to projection center
            distance = np.sqrt((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2)

            # Update if this is the closest center so far
            if distance < min_distances[pixel_id]:
                min_distances[pixel_id] = distance
                output_array[pixel_id] = list_idx

        start_idx = end_idx

    return output_array


@jit(nopython=True)
def create_skycell_pixel_lists_numba(tess_pix_skycell_id_remapped, num_skycells):
    """
    Create efficient pixel lists for each skycell using Numba acceleration.

    Args:
        tess_pix_skycell_id_remapped (array): Pixel to skycell mapping
        num_skycells (int): Number of skycells

    Returns:
        tuple: (flat_pixels, offsets) for efficient skycell pixel access
    """
    # Count pixels per skycell first
    counts = np.zeros(num_skycells, dtype=np.int32)
    for pixel_idx in range(len(tess_pix_skycell_id_remapped)):
        skycell_id = tess_pix_skycell_id_remapped[pixel_idx]
        if skycell_id != -1:
            counts[skycell_id] += 1

    # Calculate cumulative offsets
    offsets = np.zeros(num_skycells + 1, dtype=np.int32)
    for i in range(num_skycells):
        offsets[i + 1] = offsets[i] + counts[i]

    # Create flat array to store all pixel indices
    total_pixels = offsets[num_skycells]
    flat_pixels = np.zeros(total_pixels, dtype=np.int32)

    # Reset counts to use as current position trackers
    current_pos = np.copy(offsets[:-1])

    # Fill the flat array
    for pixel_idx in range(len(tess_pix_skycell_id_remapped)):
        skycell_id = tess_pix_skycell_id_remapped[pixel_idx]
        if skycell_id != -1:
            flat_pixels[current_pos[skycell_id]] = pixel_idx
            current_pos[skycell_id] += 1

    return flat_pixels, offsets


@jit(nopython=True)
def point_in_polygon(x, y, polygon):
    """
    Check if point (x, y) is inside polygon using ray casting algorithm.

    Args:
        x, y (float): Point coordinates
        polygon (array): Polygon vertices

    Returns:
        bool: True if point is inside polygon
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@jit(nopython=True)
def find_pixels_in_rectangles(coords_ps1_pix, ps1_shape):
    """
    Find integer pixel coordinates that lie within rectangles.

    Args:
        coords_ps1_pix (array): Shape (N, 4, 2) rectangle corner coordinates
        ps1_shape (tuple): (height, width) of PS1 image

    Returns:
        list: Arrays of pixel indices for each rectangle
    """
    height, width = ps1_shape
    result = []

    for rect_idx in range(coords_ps1_pix.shape[0]):
        # Get the 4 corners of the rectangle
        corners = coords_ps1_pix[rect_idx]

        # Find bounding box
        min_x = int(np.floor(np.min(corners[:, 0])))
        max_x = int(np.ceil(np.max(corners[:, 0])))
        min_y = int(np.floor(np.min(corners[:, 1])))
        max_y = int(np.ceil(np.max(corners[:, 1])))

        # Clip to image bounds
        min_x = max(0, min_x)
        max_x = min(width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(height - 1, max_y)

        # Collect pixels within the rectangle
        pixels_in_rect = []

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Check if point (x, y) is inside the rectangle
                if point_in_polygon(x, y, corners):
                    pixel_idx = y * width + x  # Convert 2D to 1D index
                    pixels_in_rect.append(pixel_idx)

        result.append(np.array(pixels_in_rect))

    return result


@jit(nopython=True)
def populate_array_numba(fll_1d_vec, ps1_pix_in_tess_result, tess_pix_in_skycell):
    """
    Populate output array with TESS pixel indices for each PS1 pixel.

    Args:
        fll_1d_vec (array): Output array to populate
        ps1_pix_in_tess_result (list): Lists of PS1 pixels for each TESS pixel
        tess_pix_in_skycell (array): TESS pixel indices
    """
    for i in range(len(ps1_pix_in_tess_result)):
        ps1_pix_in_tess = ps1_pix_in_tess_result[i]
        tess_ind = tess_pix_in_skycell[i]
        for j in range(len(ps1_pix_in_tess)):
            fll_1d_vec[ps1_pix_in_tess[j]] = tess_ind


# ============================================================================
# HIGH-LEVEL PROCESSING FUNCTIONS
# ============================================================================


def calculate_radec_corners(dataframe_skycells, buffer=120):
    """
    Calculate RA/Dec coordinates for corners of skycells with buffer.

    Args:
        dataframe_skycells (DataFrame): Skycell WCS information
        buffer (float): Buffer size in pixels

    Returns:
        ndarray: Corner coordinates for all skycells
    """
    return calculate_radec_corners_numba(
        buffer,
        dataframe_skycells["NAXIS1"].to_numpy(),
        dataframe_skycells["NAXIS2"].to_numpy(),
        dataframe_skycells["CRVAL1"].to_numpy(),
        dataframe_skycells["CRVAL2"].to_numpy(),
        dataframe_skycells["CRPIX1"].to_numpy(),
        dataframe_skycells["CRPIX2"].to_numpy(),
        dataframe_skycells["PC1_1"].to_numpy(),
        dataframe_skycells["PC1_2"].to_numpy(),
        dataframe_skycells["PC2_1"].to_numpy(),
        dataframe_skycells["PC2_2"].to_numpy(),
        dataframe_skycells["CDELT1"].to_numpy(),
        dataframe_skycells["CDELT2"].to_numpy(),
    )


def calculate_radec_corners_shift(dataframe_skycells, buffer_large=450, buffer_small=20, buffer_normal=200):
    return calculate_radec_corners_shift_numba(
        buffer_large,
        buffer_small,
        buffer_normal,
        dataframe_skycells["x"].to_numpy(),
        dataframe_skycells["y"].to_numpy(),
        dataframe_skycells["NAXIS1"].to_numpy(),
        dataframe_skycells["NAXIS2"].to_numpy(),
        dataframe_skycells["CRVAL1"].to_numpy(),
        dataframe_skycells["CRVAL2"].to_numpy(),
        dataframe_skycells["CRPIX1"].to_numpy(),
        dataframe_skycells["CRPIX2"].to_numpy(),
        dataframe_skycells["PC1_1"].to_numpy(),
        dataframe_skycells["PC1_2"].to_numpy(),
        dataframe_skycells["PC2_1"].to_numpy(),
        dataframe_skycells["PC2_2"].to_numpy(),
        dataframe_skycells["CDELT1"].to_numpy(),
        dataframe_skycells["CDELT2"].to_numpy(),
    )


def calculate_radec_center(dataframe_skycells):
    """
    Calculate RA/Dec coordinates for centers of skycells.

    Args:
        dataframe_skycells (DataFrame): Skycell WCS information

    Returns:
        ndarray: Center coordinates for all skycells
    """
    return calculate_radec_center_numba(
        dataframe_skycells["NAXIS1"].to_numpy(),
        dataframe_skycells["NAXIS2"].to_numpy(),
        dataframe_skycells["CRVAL1"].to_numpy(),
        dataframe_skycells["CRVAL2"].to_numpy(),
        dataframe_skycells["CRPIX1"].to_numpy(),
        dataframe_skycells["CRPIX2"].to_numpy(),
        dataframe_skycells["PC1_1"].to_numpy(),
        dataframe_skycells["PC1_2"].to_numpy(),
        dataframe_skycells["PC2_1"].to_numpy(),
        dataframe_skycells["PC2_2"].to_numpy(),
        dataframe_skycells["CDELT1"].to_numpy(),
        dataframe_skycells["CDELT2"].to_numpy(),
    )


def resolve_tess_input_to_fits_path(cli_path: str | os.PathLike[str]) -> tuple[str, str | None]:
    """
    Resolve the CLI positional input to a TESS FITS path.

    If ``cli_path`` is a ``cluster_template_job.json`` (same schema as
    ``multi_offset_downsampling``), returns ``reference_ffi_path`` from that file
    and the JSON path for logging. Otherwise returns the given path as the FITS
    to open and ``None`` for the JSON path.

    Returns:
        tuple: (absolute path to FITS, job JSON path or None)
    """
    p = Path(cli_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Input path not found: {p}")
    if p.suffix.lower() != ".json":
        return str(p.resolve()), None
    payload = load_cluster_template_job_payload(p)
    ref = payload.get("reference_ffi_path")
    if not ref or not str(ref).strip():
        raise KeyError(f"{p}: cluster_template_job.json missing reference_ffi_path")
    fits_p = Path(str(ref).strip())
    if not fits_p.is_file():
        raise FileNotFoundError(f"{p}: reference_ffi_path not found: {fits_p}")
    return str(fits_p.resolve()), str(p.resolve())


def load_tess_image(tess_file):
    """
    Load TESS image and extract necessary information.

    Args:
        tess_file (str): Path to TESS FITS file

    Returns:
        tuple: (data_shape, wcs, ra_center, dec_center, header, sector, camera_id, ccd_id)
    """
    hdul = fits.open(tess_file)

    try:
        header = deepcopy(hdul[1].header)
        data = hdul[1].data
        wcs = WCS(hdul[1].header)
    except Exception:
        header = deepcopy(hdul[0].header)
        data = hdul[0].data
        wcs = WCS(hdul[0].header)

    data_shape = np.shape(data)
    ra_center, dec_center = wcs.all_pix2world(data_shape[1] / 2, data_shape[0] / 2, 0)

    sector = int(tess_file.split("/")[-1].split("-")[1][1:])
    camera = int(header["CAMERA"])  # camera_id (1-4)
    ccd = int(header["CCD"])  # ccd_id (1-4)

    hdul.close()
    return data_shape, wcs, ra_center, dec_center, header, sector, camera, ccd


def create_tess_pixel_coordinates(data_shape, oversampling_factor=1):
    """
    Create coordinate arrays for TESS pixels with optional oversampling.
        
    Args:
        data_shape (tuple): Shape of TESS image (height, width).
        oversampling_factor (int): Subdivide each pixel into NxN sub-pixels.
        
    Returns:
        tuple: (pixel_coordinates, ravelled_indices)
    """
    t_y, t_x = data_shape
    N = oversampling_factor

    os_y = t_y * N
    os_x = t_x * N

    grid_y, grid_x = np.mgrid[:os_y, :os_x]

    ty_input = (grid_y + 0.5) / N - 0.5
    tx_input = (grid_x + 0.5) / N - 0.5

    tpix_coord_input = np.column_stack([ty_input.ravel(), tx_input.ravel()])
    
    ravelled_index = np.arange(os_y * os_x, dtype=np.int32)

    return tpix_coord_input, ravelled_index


def normalize_ra_degrees(ra) -> np.ndarray:
    """Wrap right ascension (degrees) to [0, 360)."""
    return np.mod(np.asarray(ra, dtype=np.float64), 360.0)


def moc_ra_shift_degrees(ref_ra: float) -> float:
    """Return degrees to add so ref_ra lands near 180, away from the 0/360 seam."""
    return 180.0 - float(normalize_ra_degrees(np.array([ref_ra]))[0])


def shift_ras_for_moc(ra, shift_deg: float) -> np.ndarray:
    """Apply a uniform RA shift and wrap to [0, 360) for MOC libraries."""
    return normalize_ra_degrees(np.asarray(ra, dtype=np.float64) + shift_deg)


def shift_polygon_ras_for_moc(vertices: np.ndarray, shift_deg: float) -> np.ndarray:
    """Shift polygon vertex RAs into [0, 360) for MOC libraries."""
    out = np.array(vertices, dtype=np.float64, copy=True)
    out[:, :, 0] = shift_ras_for_moc(out[:, :, 0], shift_deg)
    return out


def find_relevant_skycells(skycell_wcs_df, tess_wcs, data_shape, tess_buffer=150):
    """
    Find skycells that overlap with TESS image using MOC filtering.

    Args:
        skycell_wcs_df (DataFrame): Skycell WCS information
        tess_wcs (WCS): TESS image WCS
        data_shape (tuple): TESS image shape
        tess_buffer (float): Buffer around TESS image in pixels

    Returns:
        DataFrame: Filtered skycells that overlap with TESS image
    """
    # Create buffered TESS footprint
    tess_ffi_corner = tess_wcs.all_pix2world(
        np.array(
            [
                [-tess_buffer, 0],
                [-tess_buffer, data_shape[0]],
                [0, data_shape[0] + tess_buffer],
                [data_shape[1], data_shape[0] + tess_buffer],
                [data_shape[1] + tess_buffer, data_shape[0]],
                [data_shape[1] + tess_buffer, 0],
                [data_shape[1], -tess_buffer],
                [0, -tess_buffer],
            ]
        ),
        0,
    )

    footprint_ra = normalize_ra_degrees(tess_ffi_corner[:, 0])
    ra_shift = moc_ra_shift_degrees(float(np.median(footprint_ra)))
    footprint_ra = shift_ras_for_moc(footprint_ra, ra_shift)
    tess_ffi_skycoord = SkyCoord(
        ra=footprint_ra * u.deg, dec=tess_ffi_corner[:, 1] * u.deg, frame="icrs"
    )

    tess_ffi_moc = MOC.from_polygon_skycoord(tess_ffi_skycoord, complement=False, max_depth=21)
    skycell_ra = shift_ras_for_moc(skycell_wcs_df["RA"].values, ra_shift)
    sc_mask = tess_ffi_moc.contains_lonlat(skycell_ra * u.degree, skycell_wcs_df["DEC"].values * u.degree)

    return skycell_wcs_df[sc_mask].reset_index(drop=True)


def process_tess_to_skycell_mapping(tess_wcs, data_shape, tpix_coord_input, complete_wcs_skycells, edge_buffer_large=410, edge_buffer_small=70, buffer=200, n_threads=8):
    """
    Create optimized mapping from TESS pixels to skycells.

    Args:
        tess_wcs (WCS): TESS image WCS
        data_shape (tuple): TESS image shape
        tpix_coord_input (array): TESS pixel coordinates
        complete_wcs_skycells (DataFrame): Relevant skycells
        buffer (float): Buffer size for skycell edges
        n_threads (int): Number of threads for MOC processing

    Returns:
        tuple: (selected_skycells, tess_pixel_mapping)
    """
    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

    # Calculate projection centers and skycell corners
    tess_proj_center_x, tess_proj_center_y = world_ra_dec_to_pixel(
        tess_wcs,
        complete_wcs_skycells["CRVAL1"].to_numpy(),
        complete_wcs_skycells["CRVAL2"].to_numpy(),
    )
    sc_corners = calculate_radec_corners(complete_wcs_skycells, 50)
    complete_wcs_skycells["RA_Corner1"] = sc_corners[:, 0, 0]
    complete_wcs_skycells["DEC_Corner1"] = sc_corners[:, 0, 1]
    complete_wcs_skycells["RA_Corner2"] = sc_corners[:, 1, 0]
    complete_wcs_skycells["DEC_Corner2"] = sc_corners[:, 1, 1]
    complete_wcs_skycells["RA_Corner3"] = sc_corners[:, 2, 0]
    complete_wcs_skycells["DEC_Corner3"] = sc_corners[:, 2, 1]
    complete_wcs_skycells["RA_Corner4"] = sc_corners[:, 3, 0]
    complete_wcs_skycells["DEC_Corner4"] = sc_corners[:, 3, 1]

    if "projection" not in complete_wcs_skycells.columns:
        parsed = complete_wcs_skycells["NAME"].apply(get_projection_cell_id).tolist()
        cols = pd.DataFrame(parsed, columns=["projection", "cell", "y", "x"])
        complete_wcs_skycells[["projection", "y", "x", "cell"]] = cols[["projection", "y", "x", "cell"]]

    enc_sc_vertices = calculate_radec_corners_shift(complete_wcs_skycells, edge_buffer_large, edge_buffer_small, buffer)
    ref_ra = float(np.median(normalize_ra_degrees(enc_sc_vertices[:, :, 0])))
    ra_shift = moc_ra_shift_degrees(ref_ra)
    enc_sc_vertices = shift_polygon_ras_for_moc(enc_sc_vertices, ra_shift)
    # enc_sc_vertices_noedge = calculate_radec_corners_shift(complete_wcs_skycells, edge_buffer_large, buffer)

    # Get TESS pixel RA/Dec coordinates
    _x_tess = tpix_coord_input[:, 1]
    _y_tess = tpix_coord_input[:, 0]
    print(f"  Converting {len(_x_tess)} coordinates to RA/Dec...")
    _ra_tess, _dec_tess = tess_wcs.all_pix2world(_x_tess, _y_tess, 0)
    _ra_tess = shift_ras_for_moc(_ra_tess, ra_shift)
    print(f"  RA/Dec conversion complete. Running MOC filtering...")

    # Use MOC filtering for efficient polygon-point matching
    rust_result = MOC.filter_points_in_polygons(polygons=enc_sc_vertices, pix_ras=_ra_tess, pix_decs=_dec_tess, buffer=0.5, max_depth=21, n_threads=n_threads)
    print(f"  MOC filtering complete.")
    # rust_result_noedge = MOC.filter_points_in_polygons(polygons=enc_sc_vertices_noedge, pix_ras=_ra_tess, pix_decs=_dec_tess, buffer=0.5, max_depth=21, n_threads=n_threads)

    # Create efficient pixel-to-skycell mapping
    rust_result_flat = np.concatenate([arr for arr in rust_result if len(arr) > 0])
    rust_result_lengths = np.array([len(arr) for arr in rust_result])

    # Use actual number of coordinates (accounts for oversampling)
    total_coords = len(tpix_coord_input)
    tess_pix_skycell_id = create_closest_center_array_numba(rust_result_flat, rust_result_lengths, tess_proj_center_x, tess_proj_center_y, tpix_coord_input, total_coords)

    # rust_result_noedge_flat = np.concatenate([arr for arr in rust_result_noedge if len(arr) > 0])
    # rust_result_noedge_lengths = np.array([len(arr) for arr in rust_result_noedge])

    # tess_pix_skycell_id_no_edge = create_closest_center_array_numba(rust_result_noedge_flat, rust_result_noedge_lengths, tess_proj_center_x, tess_proj_center_y, tpix_coord_input, data_shape[0] * data_shape[1])

    # tess_pix_skycell_id = tess_pix_skycell_id_no_edge
    # tess_pix_skycell_id[tess_pix_skycell_id == -1] = tess_pix_skycell_id_all[tess_pix_skycell_id == -1]

    # Remap skycell IDs to consecutive integers
    unique_ids = np.unique(tess_pix_skycell_id[tess_pix_skycell_id != -1])
    id_mapping = np.full(np.max(unique_ids) + 1, -1, dtype=np.int32)
    id_mapping[unique_ids] = np.arange(len(unique_ids), dtype=np.int32)

    mask = tess_pix_skycell_id != -1
    tess_pix_skycell_id_remapped = np.full_like(tess_pix_skycell_id, -1)
    tess_pix_skycell_id_remapped[mask] = id_mapping[tess_pix_skycell_id[mask]]

    # Create selected skycells dataframe with pixel lists
    selected_skycells = complete_wcs_skycells.loc[unique_ids].reset_index(drop=True)

    flat_pixels, offsets = create_skycell_pixel_lists_numba(tess_pix_skycell_id_remapped, len(selected_skycells))

    skycell_pixel_arrays = []
    skycell_pixel_arrays_num_pix = np.zeros(len(selected_skycells), dtype=np.int32)
    for i in range(len(selected_skycells)):
        start_idx = offsets[i]
        end_idx = offsets[i + 1]
        skycell_pixel_arrays.append(flat_pixels[start_idx:end_idx])
        skycell_pixel_arrays_num_pix[i] = end_idx - start_idx

    selected_skycells["pixel_indices"] = skycell_pixel_arrays
    selected_skycells["pixel_indices_num_pix"] = skycell_pixel_arrays_num_pix

    return selected_skycells, tess_pix_skycell_id_remapped


def get_ps1_wcs_information(skycell_data):
    """
    Get WCS information from skycell data (Series, DataFrame row, or dict).

    Args:
        skycell_data (Series): Skycell WCS data

    Returns:
        tuple: (ps1_header, ps1_wcs, ps1_data_shape)
    """

    relevant_keys = ["NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2", "RADESYS", "CTYPE1", "CTYPE2"]
    # only keep relevant keys
    header_dict = skycell_data[relevant_keys].to_dict()

    ps1_header = fits.Header(header_dict)
    ps1_data_shape = (int(header_dict["NAXIS2"]), int(header_dict["NAXIS1"]))

    temp_wcs = WCS(ps1_header)

    return ps1_header, temp_wcs, ps1_data_shape


def process_skycell_pixel_mapping(tess_wcs, tpix_coord_input, ps1_wcs, ps1_data_shape, tess_pix_in_skycell, oversampling_factor=1):
    """
    Process mapping between TESS pixels and PS1 pixels for a specific skycell.

    Args:
        tess_wcs (WCS): TESS image WCS
        tpix_coord_input (array): TESS pixel coordinates (or sub-pixel coordinates if oversampled)
        ps1_wcs (WCS): PS1 skycell WCS object
        ps1_data_shape (tuple): PS1 skycell image dimensions
        tess_pix_in_skycell (array): TESS pixel/sub-pixel indices in this skycell
        oversampling_factor (int): Oversampling factor (default: 1)

    Returns:
        ndarray: Mapping array from PS1 pixels to TESS pixel/sub-pixel indices
    """
    # Get TESS pixel coordinates for this skycell
    coords = tpix_coord_input[tess_pix_in_skycell]
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]

    # Calculate footprint size based on oversampling
    footprint_halfsize = 0.5 / oversampling_factor

    # Calculate TESS pixel corners in world coordinates
    corners = np.array([np.column_stack([x_coords - footprint_halfsize, y_coords - footprint_halfsize]), np.column_stack([x_coords + footprint_halfsize, y_coords - footprint_halfsize]), np.column_stack([x_coords + footprint_halfsize, y_coords + footprint_halfsize]), np.column_stack([x_coords - footprint_halfsize, y_coords + footprint_halfsize])])  # lower_left  # lower_right  # upper_right  # upper_left

    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

    corners_reshaped = corners.transpose(1, 0, 2).reshape(-1, 2)
    world_coords = tess_wcs.all_pix2world(corners_reshaped, 0)

    # Convert to PS1 pixel coordinates
    ps1_x, ps1_y = world_ra_dec_to_pixel(ps1_wcs, world_coords[:, 0], world_coords[:, 1])
    coords_ps1_pix = np.column_stack([ps1_x, ps1_y]).reshape(len(tess_pix_in_skycell), 4, 2)

    # Find PS1 pixels within TESS pixel rectangles
    ps1_pix_in_tess_result = find_pixels_in_rectangles(coords_ps1_pix, ps1_data_shape)

    # Create output mapping array
    fll_1d_vec = np.full((ps1_data_shape[0] * ps1_data_shape[1]), -1, dtype=np.int32)
    populate_array_numba(fll_1d_vec, ps1_pix_in_tess_result, tess_pix_in_skycell)

    return fll_1d_vec.reshape(ps1_data_shape[0], ps1_data_shape[1])


def create_master_fits_header(tess_header, file_name):
    """
    Create a master FITS header for the output file.

    Args:
        tess_header (Header): Original TESS header
        file_name (str): Name of the output file

    Returns:
        Header: Processed FITS header
    """
    tess_header_master = deepcopy(tess_header)
    date_mod = datetime.now().strftime("%Y-%m-%d")

    tess_header_master["TESS_FFI"] = file_name
    tess_header_master["DATE-MOD"] = date_mod
    tess_header_master["SOFTWARE"] = "SynDiff"
    tess_header_master["CREATOR"] = "PanCAKES_v2"

    return tess_header_master


def create_fits_header(tess_header, skycell_name=None):
    """
    Create standardized FITS header for output files.

    Args:
        tess_header (Header): Original TESS header
        skycell_name (str, optional): Skycell name to include in header

    Returns:
        Header: Processed FITS header
    """
    dict_for_header = {}
    date_mod = datetime.now().strftime("%Y-%m-%d")

    cols = ["SECTOR", "CAMERA", "CCD", "TELESCOP", "INSTRUME"]
    defaults = ["N/A", 1, 1, "Not specified", "Not specified"]

    for c in range(len(cols)):
        col = cols[c]
        try:
            dict_for_header[col] = tess_header[col]
        except Exception:
            dict_for_header[col] = defaults[c]

    dict_for_header["DATE-MOD"] = date_mod
    dict_for_header["SOFTWARE"] = "SynDiff"
    dict_for_header["CREATOR"] = "PanCAKES_v2"

    if skycell_name:
        dict_for_header["SKYCELL"] = skycell_name

    return fits.Header(dict_for_header)


def save_skycell_mapping(mapping_array, skycell_name, tess_header, ps1_header, output_path, sector, camera_id, ccd_id, overwrite=True, oversampling_factor=1):
    """
    Save PS1-to-TESS pixel mapping as compressed FITS file.

    Args:
        mapping_array (ndarray): 2D mapping array
        skycell_name (str): Skycell name
        tess_header (Header): TESS image header
        ps1_header (Header): PS1 skycell header
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        camera_id (int): TESS camera identifier
        ccd_id (int): TESS CCD identifier
        overwrite (bool): Whether to overwrite existing files
        oversampling_factor (int): Oversampling factor (default: 1)
    """

    # Add oversampling suffix if needed
    if oversampling_factor > 1:
        file_name = f"tess_s{sector}_{camera_id}_{ccd_id}_{skycell_name}_os{oversampling_factor}.fits"
    else:
        file_name = f"tess_s{sector}_{camera_id}_{ccd_id}_{skycell_name}.fits"

    # Create headers
    base_header = create_fits_header(tess_header, skycell_name)
    
    # Add oversampling info to header
    if oversampling_factor > 1:
        base_header["OVERSAMP"] = (oversampling_factor, "Oversampling factor (NxN sub-pixels per TESS pixel)")
        base_header["OSAMPRES"] = (f"{2048 * oversampling_factor}x{2048 * oversampling_factor}", "Effective oversampled resolution")

    new_fits_header = fits.Header()
    new_fits_header["SIMPLE"] = "T"
    new_fits_header += base_header
    new_fits_header_extended = new_fits_header + ps1_header

    # Process mapping array
    mapping_array[mapping_array == -1] = -1  # Ensure -1 for unmapped pixels

    # Create FITS file with oversampling subdirectory if needed
    if oversampling_factor > 1:
        file_path = os.path.join(output_path, f"oversampling_{oversampling_factor}", f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}", file_name)
    else:
        file_path = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    primary_hdu = fits.PrimaryHDU(header=new_fits_header)
    image_hdu = fits.ImageHDU(data=np.int64(mapping_array), header=new_fits_header_extended)
    image_hdu.scale("int32", bscale=1.0, bzero=32768.0)
    image_hdu.header["EXTNAME"] = "TESS_PIXEL_MAP"
    image_hdu.header["BSCALE"] = 1.0
    image_hdu.header["BZERO"] = 32768.0

    hdul = fits.HDUList([primary_hdu, image_hdu])
    hdul.verify("fix")
    hdul.writeto(file_path, overwrite=overwrite)

    # Compress file
    compress_cmd = f"gzip -f {file_path}"
    os.system(compress_cmd)


def master_skycells_csv_paths(output_path, sector, camera_id, ccd_id, oversampling_factor=1):
    """Return (partial_csv_path, final_csv_path) for mapping stage outputs."""
    if oversampling_factor > 1:
        file_stem = (
            f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_skycells_list_os{oversampling_factor}"
        )
        base_path = os.path.join(
            output_path,
            f"oversampling_{oversampling_factor}",
            f"sector_{sector:04d}",
            f"camera_{camera_id}",
            f"ccd_{ccd_id}",
        )
    else:
        file_stem = f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_skycells_list"
        base_path = os.path.join(
            output_path,
            f"sector_{sector:04d}",
            f"camera_{camera_id}",
            f"ccd_{ccd_id}",
        )
    return (
        os.path.join(base_path, f"{file_stem}.partial.csv"),
        os.path.join(base_path, f"{file_stem}.csv"),
    )


def prepare_mapping_csv_workspace(
    output_path, sector, camera_id, ccd_id, overwrite=True, oversampling_factor=1
):
    """Remove stale mapping CSV artifacts before starting a new run."""
    partial_csv, final_csv = master_skycells_csv_paths(
        output_path, sector, camera_id, ccd_id, oversampling_factor
    )
    os.makedirs(os.path.dirname(partial_csv), exist_ok=True)
    if os.path.isfile(partial_csv):
        os.remove(partial_csv)
    if overwrite and os.path.isfile(final_csv):
        os.remove(final_csv)


def finalize_master_skycells_csv(output_path, sector, camera_id, ccd_id, oversampling_factor=1):
    """Publish the completed master skycells CSV by renaming the partial file."""
    partial_csv, final_csv = master_skycells_csv_paths(
        output_path, sector, camera_id, ccd_id, oversampling_factor
    )
    if not os.path.isfile(partial_csv):
        raise FileNotFoundError(f"Partial master skycells CSV missing: {partial_csv}")
    os.replace(partial_csv, final_csv)
    print(f"Master skycell CSV saved to: {final_csv}")
    return final_csv


def save_master_mapping(tess_pix_skycell_mapping, selected_skycells, ffi_file_name, tess_header, data_shape, output_path, sector, camera_id, ccd_id, overwrite=True, oversampling_factor=1):
    """
    Save master TESS-to-skycell mapping file.

    Args:
        tess_pix_skycell_mapping (ndarray): TESS pixel to skycell mapping
        selected_skycells (DataFrame): Selected skycells information
        tess_header (Header): TESS image header
        data_shape (tuple): TESS image shape (original, before oversampling)
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        camera_id (int): TESS camera identifier
        ccd_id (int): TESS CCD identifier
        overwrite (bool): Whether to overwrite existing files
        oversampling_factor (int): Oversampling factor (default: 1)
    """
    # Create filename with oversampling suffix if needed
    if oversampling_factor > 1:
        file_name = f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_pixels2skycells_os{oversampling_factor}.fits"
        base_path = os.path.join(output_path, f"oversampling_{oversampling_factor}", f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}")
    else:
        file_name = f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_pixels2skycells.fits"
        base_path = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}")

    partial_csv, _final_csv = master_skycells_csv_paths(
        output_path, sector, camera_id, ccd_id, oversampling_factor
    )
    os.makedirs(os.path.dirname(partial_csv), exist_ok=True)
    selected_skycells.to_csv(partial_csv)

    # Create header
    master_header = create_master_fits_header(tess_header, ffi_file_name)
    master_header["DATE-MOD"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    
    # Add oversampling info to header
    if oversampling_factor > 1:
        master_header["OVERSAMP"] = (oversampling_factor, "Oversampling factor (NxN sub-pixels per TESS pixel)")
        master_header["OSAMPRES"] = (f"{data_shape[0] * oversampling_factor}x{data_shape[1] * oversampling_factor}", "Effective oversampled resolution")
        master_header["NAXIS1"] = data_shape[1] * oversampling_factor
        master_header["NAXIS2"] = data_shape[0] * oversampling_factor

    # Create FITS file
    file_path = os.path.join(base_path, file_name)
    primary_hdu = fits.PrimaryHDU()

    # Reshape mapping to 2D with oversampling dimensions
    if oversampling_factor > 1:
        oversampled_shape = (data_shape[0] * oversampling_factor, data_shape[1] * oversampling_factor)
        mapping_2d = tess_pix_skycell_mapping.reshape(oversampled_shape)
    else:
        mapping_2d = tess_pix_skycell_mapping.reshape(data_shape)
    image_hdu = fits.ImageHDU(data=np.int16(mapping_2d), header=master_header)

    # Create skycell table
    table = fits.BinTableHDU.from_columns([fits.Column(name="SKYCELL", format="20A", array=selected_skycells["NAME"].values), fits.Column(name="SKYCIND", format="K", array=np.arange(len(selected_skycells)))])

    hdul = fits.HDUList([primary_hdu, image_hdu, table])
    hdul.writeto(file_path, overwrite=overwrite)

    # Compress file
    compress_cmd = f"gzip -f {file_path}"
    os.system(compress_cmd)


def process_single_skycell(args):
    """
    Process a single skycell mapping task for parallel execution.

    Args:
        args (tuple): Contains (skycell_row, tess_wcs, tpix_coord_input,
                      tess_header, output_path, sector, camera_id, ccd_id, overwrite, all_skycells, oversampling_factor)

    Returns:
        tuple: (success, skycell_name, padding_info, error_message)
    """
    try:
        (skycell_row, tess_wcs, tpix_coord_input, tess_header, output_path, sector, camera_id, ccd_id, pad_distance, edge_exclusion, overwrite, all_skycells, oversampling_factor) = args

        skycell_name = skycell_row["NAME"]
        tess_pix_in_skycell = skycell_row["pixel_indices"]

        if len(tess_pix_in_skycell) == 0:
            return (False, skycell_name, {}, "No TESS pixels in skycell")

        # Get PS1 header and WCS information directly from skycell_row
        ps1_header, ps1_wcs, ps1_data_shape = get_ps1_wcs_information(skycell_row)

        # Process skycell pixel mapping
        mapping_array = process_skycell_pixel_mapping(tess_wcs, tpix_coord_input, ps1_wcs, ps1_data_shape, tess_pix_in_skycell, oversampling_factor)

        # Analyze padding requirements immediately while we have the mapping array
        padding_info = analyze_single_skycell_padding(skycell_name, mapping_array, ps1_wcs, skycell_row, all_skycells, pad_distance=pad_distance, edge_exclusion=edge_exclusion)

        # Save mapping
        save_skycell_mapping(mapping_array, skycell_name, tess_header, ps1_header, output_path, sector, camera_id, ccd_id, overwrite, oversampling_factor)

        return (True, skycell_name, padding_info, None)

    except Exception as e:
        return (False, skycell_name, {}, str(e))


def analyze_single_skycell_padding(skycell_name, mapping_array, ps1_wcs, skycell_row, all_skycells, pad_distance=500, edge_exclusion=10):
    """
    Analyze padding requirements for a single skycell.

    Args:
        skycell_name (str): Name of the skycell
        mapping_array (ndarray): 2D mapping array for this skycell
        skycell_row (Series, optional): Row data for the skycell, needed for special padding
        all_skycells (DataFrame, optional): DataFrame with all skycells, needed for special padding
        pad_distance (int): Distance from edge to check for padding requirements
        edge_exclusion (int): Pixels to exclude from very edge

    Returns:
        dict: Dictionary with padding information for this skycell
    """
    # Initialize padding info with empty values
    padding_directions = ["top", "right", "top_right", "bottom", "left", "bottom_left", "bottom_right", "top_left"]
    padding_info = {f"pad_skycell_{direction}": "" for direction in padding_directions}
    padding_info.update({"special_padding_needed": False, "edge_pixels_used": False, "good_side_fail": False, "special_padding_flags": [False] * 8})

    # Parse skycell coordinates
    projection, cell, y, x = get_projection_cell_id(skycell_name)

    # Check for edge pixel usage (stricter check)
    check_no_edge_bad_pix = check_tess_mapping_padding(mapping_array, pad_distance=10, edge_exclusion=0)
    if check_no_edge_bad_pix.any_needed():
        padding_info["edge_pixels_used"] = True
        print(f"Warning: Edge pixels are being used for {skycell_name}. This is not good")

    # Check padding requirements
    pad_requirements = check_tess_mapping_padding(mapping_array, pad_distance=pad_distance, edge_exclusion=edge_exclusion)

    # Check if good sides fail
    if pad_requirements.good_side_fail():
        if x < 9 and y < 9 and x > 0 and y > 0:
            padding_info["good_side_fail"] = True
        else:
            if x == 0 and y == 0:
                pass
            elif y == 0:
                if not (pad_requirements.bottom or pad_requirements.bottom_left or pad_requirements.bottom_right):
                    padding_info["good_side_fail"] = True
            elif x == 0:
                if not (pad_requirements.left or pad_requirements.top_left or pad_requirements.bottom_left):
                    padding_info["good_side_fail"] = True
            elif y == 9:
                if not (pad_requirements.top or pad_requirements.top_left or pad_requirements.top_right):
                    padding_info["good_side_fail"] = True
            elif x == 9:
                if not (pad_requirements.right or pad_requirements.top_right or pad_requirements.bottom_right):
                    padding_info["good_side_fail"] = True

    # Track which directions need special padding from other projections
    special_padding_diff_projection = np.zeros(8, dtype=bool)  # top, right, top_right, bottom, left, bottom_left, bottom_right, top_left
    padding_list = pad_requirements.to_list()

    # Get padding directions
    padding_directions = ["top", "right", "top_right", "bottom", "left", "bottom_left", "bottom_right", "top_left"]

    # Get neighboring cells based on position in the grid
    for i, direction in enumerate(padding_directions):
        if padding_list[i]:
            # Default padding approach (within same projection)
            if direction == "top" and y < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y + 1}{x}"
            elif direction == "right" and x < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y}{x + 1}"
            elif direction == "top_right" and y < 9 and x < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y + 1}{x + 1}"
            elif direction == "bottom" and y > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y - 1}{x}"
            elif direction == "left" and x > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y}{x - 1}"
            elif direction == "bottom_left" and y > 0 and x > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y - 1}{x - 1}"
            elif direction == "bottom_right" and y > 0 and x < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y - 1}{x + 1}"
            elif direction == "top_left" and y < 9 and x > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y + 1}{x - 1}"
            else:
                # This side needs special padding (from different projection)
                special_padding_diff_projection[i] = True

    # If we need special padding and have the necessary inputs
    if np.any(special_padding_diff_projection) and skycell_row is not None and all_skycells is not None:
        padding_info["special_padding_needed"] = True
        padding_info["special_padding_flags"] = special_padding_diff_projection.tolist()

        # Process each direction that needs special padding
        for i, direction in enumerate(padding_directions):
            if special_padding_diff_projection[i]:
                try:
                    # Get padding corners for this direction
                    padding_corners = get_padding_corners(skycell_row, ps1_wcs, direction, pad_size=pad_distance, edge_exclusion=edge_exclusion)

                    # Find the best padding skycell
                    result = find_best_padding_skycell(skycell_row, padding_corners, all_skycells)

                    # Update padding info based on result
                    if result["status"] == "full_coverage":
                        # Single skycell with full coverage
                        padding_info[f"pad_skycell_{direction}"] = result["best_match"]
                    elif result["status"] == "combined_coverage" and result["combined_solutions"]:
                        # Combined solution (take the first one)
                        solution = result["combined_solutions"][0]
                        padding_info[f"pad_skycell_{direction}"] = "/".join(solution["skycells"])
                    elif result["status"] == "partial_coverage":
                        # Best partial coverage
                        padding_info[f"pad_skycell_{direction}"] = result["best_match"]
                except Exception as e:
                    print(f"Error processing {skycell_name} {direction} padding: {e}")
                    padding_info[f"pad_skycell_{direction}"] = "None"

    return padding_info


def _padding_value_for_csv(key: str, value) -> str:
    """Coerce padding metadata to CSV-safe strings (pandas StringDtype rejects bools)."""
    if key == "special_padding_flags" or isinstance(value, (list, tuple, np.ndarray)):
        seq = list(value) if isinstance(value, np.ndarray) else list(value)
        return str(seq)
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return ""
    return value


def update_skycells_with_padding_info(selected_skycells, padding_results):
    """
    Update selected_skycells dataframe with padding information collected from worker threads.

    Args:
        selected_skycells (DataFrame): DataFrame with skycell information
        padding_results (dict): Dictionary mapping skycell names to padding info

    Returns:
        DataFrame: Updated dataframe with padding information
    """

    # Initialize new columns for padding information
    padding_columns = [
        "pad_skycell_top",
        "pad_skycell_right",
        "pad_skycell_top_right",
        "pad_skycell_bottom",
        "pad_skycell_left",
        "pad_skycell_bottom_left",
        "pad_skycell_bottom_right",
        "pad_skycell_top_left",
        "special_padding_needed",
        "edge_pixels_used",
        "good_side_fail",
        "special_padding_flags",
    ]
    bool_padding_columns = {"special_padding_needed", "edge_pixels_used", "good_side_fail"}

    for col in padding_columns:
        if col not in selected_skycells.columns:
            if col == "special_padding_flags":
                selected_skycells[col] = str([False] * 8)
            elif col in bool_padding_columns:
                selected_skycells[col] = "False"
            else:
                selected_skycells[col] = ""

    # Parse projection, cell, y, x for all skycells if not already done
    if "projection" not in selected_skycells.columns:
        parsed = selected_skycells["NAME"].apply(get_projection_cell_id).tolist()
        cols = pd.DataFrame(parsed, columns=["projection", "cell", "y", "x"])
        selected_skycells[["projection", "y", "x", "cell"]] = cols[["projection", "y", "x", "cell"]]

    # Update dataframe with padding information from worker results
    for idx, skycell_row in selected_skycells.iterrows():
        skycell_name = skycell_row["NAME"]

        if skycell_name in padding_results:
            padding_info = padding_results[skycell_name]

            # Update dataframe with padding information
            for key, value in padding_info.items():
                if key not in selected_skycells.columns:
                    continue
                selected_skycells.at[idx, key] = _padding_value_for_csv(key, value)

    return selected_skycells


def save_updated_skycell_csv(selected_skycells, output_path, sector, camera_id, ccd_id, oversampling_factor=1):
    """
    Save updated CSV file with padding information for processed skycells.

    Args:
        selected_skycells (DataFrame): Processed skycells with padding info
        original_skycell_wcs_df (DataFrame): Original skycell WCS dataframe
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        camera_id (int): TESS camera identifier
        ccd_id (int): TESS CCD identifier
        oversampling_factor (int): Oversampling factor (default: 1)
    """
    partial_csv, _final_csv = master_skycells_csv_paths(
        output_path, sector, camera_id, ccd_id, oversampling_factor
    )
    os.makedirs(os.path.dirname(partial_csv), exist_ok=True)
    selected_skycells.to_csv(partial_csv, index=False)
    print(f"Updated partial skycell CSV saved to: {partial_csv}")


# ============================================================================
# GAIA CATALOG DOWNLOAD UTILITIES
# ============================================================================

_gaia_logged_in = False
_gaia_module = None


def _get_gaia():
    """Import astroquery Gaia client on first use (avoids CLI startup banner)."""
    global _gaia_module
    if _gaia_module is None:
        from astroquery.gaia import Gaia

        _gaia_module = Gaia
    return _gaia_module


def resolve_gaia_credentials_path(cli_path=None):
    """
    Resolve path to Gaia TAP+ credentials file (username line 1, password line 2).

    Returns None for anonymous TAP access when no explicit path is given.

    Raises FileNotFoundError if an explicit path was given but missing.
    """
    if cli_path is not None and str(cli_path).strip():
        p = Path(os.path.expanduser(str(cli_path).strip())).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Gaia credentials file not found: {p}")
        return str(p)
    return None


def ensure_gaia_login(gaia_credentials_file=None):
    """Call ``Gaia.login(credentials_file=...)`` once per process when a path resolves."""
    global _gaia_logged_in
    if _gaia_logged_in:
        return
    path = resolve_gaia_credentials_path(gaia_credentials_file)
    if not path:
        return
    _get_gaia().login(credentials_file=path)
    _gaia_logged_in = True


def _gaia_transient_error(exc):
    msg = str(exc).lower()
    return "408" in msg or "timeout" in msg or "aborted" in msg


def _emit_gaia_query_for_manual_fallback(catalog_query, adql_out_path=None):
    """
    After a failed TAP query, print the ADQL for the Gaia Archive web UI and optionally save it.

    Users can open https://gea.esac.esa.int/archive/ , sign in, use the ADQL tab,
    paste the query, and download results (e.g. CSV) to replace the pipeline catalog.
    """
    q = catalog_query.strip()
    print("\n" + "=" * 72)
    print("Gaia query failed via astroquery. Run this ADQL manually in the Gaia Archive:")
    print("  https://gea.esac.esa.int/archive/  (ADQL / Advanced query)")
    print("Use async or sync as appropriate; large results may need an async job.")
    print("-" * 72)
    print(q)
    print("=" * 72 + "\n")
    if adql_out_path:
        try:
            with open(adql_out_path, "w", encoding="utf-8") as f:
                f.write(q + "\n")
            print(f"ADQL saved for copy/paste: {adql_out_path}")
        except OSError as io_err:
            print(f"(Could not write ADQL file {adql_out_path}: {io_err})")


def _run_gaia_catalog_query_async(catalog_query, max_attempts=4, initial_delay_s=30.0):
    """Run ``launch_job_async`` + ``get_results`` with retries on transient Gaia failures."""
    delay = initial_delay_s
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            catalog_job = _get_gaia().launch_job_async(catalog_query)
            return catalog_job.get_results()
        except Exception as e:
            last_exc = e
            if attempt >= max_attempts or not _gaia_transient_error(e):
                raise
            print(f"⚠️ Gaia catalog query attempt {attempt}/{max_attempts} failed ({e}); retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2.0, 300.0)
    raise last_exc


def download_gaia_catalog(
    tess_wcs,
    data_shape,
    output_path,
    sector,
    camera_id,
    ccd_id,
    pixel_padding=50,
    magnitude_limit=18,
    gaia_credentials_file=None,
):
    """
    Download Gaia catalog for the FFI footprint with padding.

    Args:
        tess_wcs (WCS): TESS WCS object
        data_shape (tuple): Shape of TESS image (height, width)
        output_path (str): Base output directory
        sector (int): TESS sector number
        camera_id (int): Camera ID
        ccd_id (int): CCD ID
        pixel_padding (int): Number of pixels to pad around FFI footprint (default: 50)
        magnitude_limit (float): Magnitude limit for Gaia RP band (default: 18)
        gaia_credentials_file (str, optional): Explicit credentials path; None uses env / default sibling file.

    Returns:
        str: Path to the saved catalog CSV file
    """
    ensure_gaia_login(gaia_credentials_file)
    print(f"📡 Downloading Gaia catalog for FFI with {pixel_padding} pixel padding...")

    # Create output directory structure
    catalog_dir = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}")
    os.makedirs(catalog_dir, exist_ok=True)

    catalog_filename = f"gaia_catalog_s{sector:04d}_{camera_id}_{ccd_id}.csv"
    catalog_path = os.path.join(catalog_dir, catalog_filename)

    # Get image dimensions for the footprint calculation
    height, width = data_shape

    # Create padded corners for footprint calculation
    # Add padding to corners (expand outward)
    padded_corners = np.array(
        [
            [-pixel_padding, -pixel_padding],  # bottom-left
            [width - 1 + pixel_padding, -pixel_padding],  # bottom-right
            [width - 1 + pixel_padding, height - 1 + pixel_padding],  # top-right
            [-pixel_padding, height - 1 + pixel_padding],  # top-left
        ]
    )

    # Convert padded pixel coordinates to world coordinates
    sky_coords = tess_wcs.pixel_to_world(padded_corners[:, 0], padded_corners[:, 1])

    print(f"FFI footprint with {pixel_padding} pixel padding calculated")

    # Extract RA and Dec coordinates
    ra_coords = sky_coords.ra.deg
    dec_coords = sky_coords.dec.deg

    # Format footprint for ADQL query - interleave RA and Dec coordinates
    footprint_coords = []
    for ra, dec in zip(ra_coords, dec_coords):
        footprint_coords.extend([ra, dec])

    polygon_str = ",".join(map(str, footprint_coords))

    print("📚 Querying Gaia for catalog (positions, magnitudes, and errors)...")

    # Single async query (avoids a separate COUNT job that doubled load and timeout risk).
    catalog_query = f"""
    SELECT
        source_id, ra, ra_error, dec, dec_error,
        parallax, parallax_error,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag
    FROM
        gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        POLYGON('ICRS', {polygon_str})
    )
    AND phot_rp_mean_mag < {magnitude_limit}
    """

    print("Submitting job to Gaia... this may take a few minutes.")

    adql_sidecar = os.path.splitext(catalog_path)[0] + "_manual_query.adql"
    try:
        gaia_catalog = _run_gaia_catalog_query_async(catalog_query)
    except Exception as e:
        print(f"❌ ERROR: Gaia catalog query failed. Error: {e}")
        _emit_gaia_query_for_manual_fallback(catalog_query, adql_out_path=adql_sidecar)
        raise

    # Convert to pandas DataFrame and save as CSV
    df = gaia_catalog.to_pandas()
    df.to_csv(catalog_path, index=False)

    print(f"✅ Gaia catalog saved to: {catalog_path}")
    print(f"📊 Downloaded {len(df)} stars in the padded FFI area.")

    return catalog_path


# ============================================================================
# GAIA CATALOG WORKFLOW FUNCTIONS
# ============================================================================


def download_gaia_catalog_for_tess_file(
    tess_file,
    output_path,
    pixel_padding=50,
    magnitude_limit=18.0,
    force_download=False,
    gaia_credentials_file=None,
):
    """
    Standalone function to download Gaia catalog for a TESS FITS file.

    This function handles the complete workflow of loading TESS data and downloading
    the corresponding Gaia catalog with proper file organization.

    Args:
        tess_file (str): Path to TESS FITS file
        output_path (str): Base directory for Gaia catalogs (default: ./data/catalogs)
        pixel_padding (int): Number of pixels to pad around FFI footprint (default: 50)
        magnitude_limit (float): Magnitude limit for Gaia RP band (default: 18.0)
        force_download (bool): Force download even if catalog already exists (default: False)
        gaia_credentials_file (str, optional): Explicit Gaia credentials file path (see resolve_gaia_credentials_path).

    Returns:
        str: Path to the downloaded catalog CSV file
    """
    print(f"🚀 Starting Gaia catalog download workflow for: {tess_file}")

    # Load TESS data to get WCS and metadata
    print("Loading TESS image data...")
    data_shape, tess_wcs, ra_center, dec_center, tess_header, sector, camera_id, ccd_id = load_tess_image(tess_file)

    # Create output directory structure
    catalog_dir = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}")
    catalog_filename = f"gaia_catalog_s{sector:04d}_{camera_id}_{ccd_id}.csv"
    catalog_path = os.path.join(catalog_dir, catalog_filename)

    # Check if catalog already exists
    if not force_download and os.path.exists(catalog_path):
        print(f"✅ Gaia catalog already exists at: {catalog_path}")
        return catalog_path

    # Download the catalog
    print(f"📡 Downloading Gaia catalog with {pixel_padding} pixel padding...")
    try:
        catalog_path = download_gaia_catalog(
            tess_wcs=tess_wcs,
            data_shape=data_shape,
            output_path=output_path,
            sector=sector,
            camera_id=camera_id,
            ccd_id=ccd_id,
            pixel_padding=pixel_padding,
            magnitude_limit=magnitude_limit,
            gaia_credentials_file=gaia_credentials_file,
        )
        print(f"✅ Gaia catalog download complete: {catalog_path}")
        return catalog_path

    except Exception as e:
        print(f"❌ ERROR: Failed to download Gaia catalog: {e}")
        raise


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================


def process_tess_image_optimized(tess_file, skycell_wcs_csv, output_path, pad_distance=480, edge_exclusion=10, edge_buffer_large=410, edge_buffer_small=70, buffer=200, tess_buffer=150, n_threads=8, overwrite=True, max_workers=None, oversampling_factor=1):
    """
    Main optimized pipeline for processing TESS images with PanSTARRS1 skycells.

    This function implements the complete optimized workflow:
    1. Load TESS image and skycell database
    2. Find relevant skycells using MOC filtering
    3. Create optimized TESS-to-skycell mapping
    4. Process each skycell for PS1-to-TESS pixel mapping
    5. Save all mapping files

    Args:
        tess_file (str): Path to TESS FITS file
        skycell_wcs_csv (str): Path to skycell WCS CSV file
        output_path (str): Output directory for mapping files
        pad_distance (int): Pad distance in pixels for padding checks
        edge_exclusion (int): Edge exclusion in pixels for padding checks
        edge_buffer_large (int): Large edge buffer for WCS corner expansion
        edge_buffer_small (int): Small edge buffer for WCS corner expansion
        buffer (int): Buffer size for PS1 skycells in pixels
        tess_buffer (int): Buffer size for TESS image in pixels
        n_threads (int): Number of threads for parallel processing
        overwrite (bool): Whether to overwrite existing files
        max_workers (int, optional): Maximum number of parallel workers for skycell processing.
                                   If None, uses min(32, len(skycells), cpu_count + 4)
        oversampling_factor (int): Subdivide each TESS pixel into NxN sub-pixels (default: 1)

    Returns:
        dict: Processing results and statistics
    """
    start_time = time.time()
    
    # Print oversampling information if enabled
    if oversampling_factor > 1:
        print(f"Starting optimized TESS image processing with {oversampling_factor}x{oversampling_factor} oversampling...")
        if oversampling_factor > 4:
            print(f"Warning: High oversampling factor ({oversampling_factor}) may require significant memory and processing time (approximately {oversampling_factor**2}x increase)")
    else:
        print("Starting optimized TESS image processing...")

    # Load data
    print("Loading TESS image and skycell database...")
    data_shape, tess_wcs, ra_center, dec_center, tess_header, sector, camera_id, ccd_id = load_tess_image(tess_file)
    prepare_mapping_csv_workspace(
        output_path, sector, camera_id, ccd_id, overwrite, oversampling_factor
    )

    skycell_wcs_df = pd.read_csv(skycell_wcs_csv)

    # Calculate skycell centers if not present
    if "RA" not in skycell_wcs_df.columns or "DEC" not in skycell_wcs_df.columns:
        print("Calculating skycell centers...")
        sc_center_ra_dec = calculate_radec_center(skycell_wcs_df.reset_index(drop=True))
        skycell_wcs_df["RA"] = sc_center_ra_dec[:, 0]
        skycell_wcs_df["DEC"] = sc_center_ra_dec[:, 1]

    # Create TESS pixel coordinates (with oversampling if requested)
    if oversampling_factor > 1:
        print(f"Creating oversampled TESS pixel coordinate arrays ({oversampling_factor}x{oversampling_factor})...")
    else:
        print("Creating TESS pixel coordinate arrays...")
    tpix_coord_input, ravelled_index = create_tess_pixel_coordinates(data_shape, oversampling_factor)

    # Find relevant skycells
    print("Finding relevant skycells using MOC filtering...")
    complete_wcs_skycells = find_relevant_skycells(skycell_wcs_df, tess_wcs, data_shape, tess_buffer)

    if len(complete_wcs_skycells) == 0:
        print("No relevant skycells found!")
        return {"status": "error", "message": "No relevant skycells found"}

    # print(f"Found {len(complete_wcs_skycells)} relevant skycells")

    # Create TESS-to-skycell mapping
    print("Creating optimized TESS-to-skycell mapping...")
    selected_skycells, tess_pix_skycell_mapping = process_tess_to_skycell_mapping(tess_wcs, data_shape, tpix_coord_input, complete_wcs_skycells, edge_buffer_large=edge_buffer_large, edge_buffer_small=edge_buffer_small, buffer=buffer, n_threads=n_threads)

    if np.any(tess_pix_skycell_mapping == -1):
        print("Warning: Some TESS pixels are not mapped to any skycell. This may affect the results.")

    # Save master mapping
    print("Saving master TESS-to-skycell mapping...")
    ffi_file_name = os.path.basename(tess_file)
    save_master_mapping(tess_pix_skycell_mapping, selected_skycells, ffi_file_name, tess_header, data_shape, output_path, sector, camera_id, ccd_id, overwrite, oversampling_factor)
    print(f"Processing time: {(time.time() - start_time):.2f} seconds")

    # Process each skycell
    print("Processing individual skycell mappings...")
    processed_skycells = 0
    skipped_skycells = 0

    # Determine number of workers for parallel processing
    if max_workers is None:
        import multiprocessing

        max_workers = min(32, len(selected_skycells), multiprocessing.cpu_count() + 4)

    # Prepare arguments for parallel processing
    task_args = []
    for _, skycell_row in selected_skycells.iterrows():
        args = (skycell_row, tess_wcs, tpix_coord_input, tess_header, output_path, sector, camera_id, ccd_id, pad_distance, edge_exclusion, overwrite, selected_skycells, oversampling_factor)
        task_args.append(args)

    # Process skycells in parallel with progress bar
    padding_results = {}  # Store padding info for each skycell

    if len(task_args) > 0:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_skycell, args): args for args in task_args}

            # Process completed tasks with progress bar
            with tqdm(total=len(task_args), desc="Processing skycells") as pbar:
                for future in as_completed(future_to_args):
                    success, skycell_name, padding_info, error_message = future.result()

                    if success:
                        processed_skycells += 1
                        if padding_info:
                            padding_results[skycell_name] = padding_info
                    else:
                        skipped_skycells += 1
                        if error_message != "No TESS pixels in skycell":
                            print(f"Error processing skycell {skycell_name}: {error_message}")

                    pbar.update(1)

    # Update skycells with padding information collected from worker threads
    if padding_results:
        print("\nUpdating skycells with padding information...")
        selected_skycells = update_skycells_with_padding_info(selected_skycells, padding_results)

        # Save updated CSV with padding information
        save_updated_skycell_csv(selected_skycells, output_path, sector, camera_id, ccd_id, oversampling_factor)

    finalize_master_skycells_csv(
        output_path, sector, camera_id, ccd_id, oversampling_factor
    )

    total_time = time.time() - start_time

    results = {
        "status": "success",
        "tess_file": tess_file,
        "total_skycells_found": len(complete_wcs_skycells),
        "selected_skycells": len(selected_skycells),
        "processed_skycells": processed_skycells,
        "skipped_skycells": skipped_skycells,
        "processing_time_seconds": total_time,
        "data_shape": data_shape,
        "ra_center": ra_center,
        "dec_center": dec_center,
        "oversampling_factor": oversampling_factor,
    }

    print("\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processed: {processed_skycells} skycells")
    print(f"Skipped: {skipped_skycells} skycells")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a TESS FITS file and generate PS1 skycell pixel mappings.")

    # Positional required argument
    parser.add_argument(
        "tess_file",
        help="Path to the TESS FITS file, or cluster_template_job.json (uses reference_ffi_path from JSON)",
    )

    # Optional arguments (defaults chosen to match previous hard-coded values)
    parser.add_argument("--skycell_wcs_csv", default="./data/SkyCells/skycell_wcs.csv", help="Path to skycell WCS CSV file")
    parser.add_argument("--output_path", default="./data/skycell_pixel_mapping", help="Output directory for mapping files")
    parser.add_argument("--pad_distance", type=int, default=480, help="Pad distance in pixels for padding checks")
    parser.add_argument("--edge_exclusion", type=int, default=10, help="Edge exclusion in pixels for padding checks")
    parser.add_argument("--edge_buffer_large", type=int, default=410, help="Large edge buffer for WCS corner expansion")
    parser.add_argument("--edge_buffer_small", type=int, default=70, help="Small edge buffer for WCS corner expansion")
    parser.add_argument("--buffer", type=int, default=200, help="General buffer size in pixels")
    parser.add_argument("--tess_buffer", type=int, default=150, help="TESS footprint buffer in pixels used for MOC filtering")
    parser.add_argument("--n_threads", type=int, default=8, help="Number of threads to use in MOC filtering")
    parser.add_argument("--max_workers", type=int, default=None, help="Max workers for ProcessPoolExecutor (default: auto)")
    
    # Oversampling option
    parser.add_argument("--oversampling_factor", type=int, default=1, help="Subdivide each TESS pixel into NxN sub-pixels for high-resolution template (default: 1, no oversampling)")

    # Gaia catalog download options
    parser.add_argument("--skip-download-catalog", action="store_false", help="Skip Gaia catalog download for the TESS file")
    parser.add_argument("--gaia_catalog_dir", default="./data/catalogs", help="Base directory for Gaia catalogs")
    parser.add_argument("--gaia_pixel_padding", type=int, default=50, help="Pixel padding around FFI for Gaia catalog download")
    parser.add_argument("--gaia_magnitude_limit", type=float, default=18.0, help="Magnitude limit for Gaia RP band")
    parser.add_argument("--force-gaia-download", action="store_true", help="Force Gaia catalog download even if it already exists")
    parser.add_argument(
        "--gaia-credentials-file",
        default=None,
        metavar="PATH",
        help="Gaia Archive credentials file (line 1: username, line 2: password). "
        "Path to Gaia TAP+ credentials file (username line 1, password line 2).",
    )

    # Overwrite default preserved (default True). Provide flags to explicitly enable/disable.
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite existing output files (default)")
    group.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite existing output files")
    parser.set_defaults(overwrite=True, skip_download_catalog=True)

    args = parser.parse_args()

    try:
        tess_fits_path, cluster_job_json = resolve_tess_input_to_fits_path(args.tess_file)
    except (FileNotFoundError, KeyError, ValueError) as e:
        parser.error(str(e))
    if cluster_job_json:
        print(f"Using cluster job JSON {cluster_job_json} → reference_ffi_path: {tess_fits_path}")

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Handle Gaia catalog download if requested
    if args.skip_download_catalog:
        try:
            catalog_path = download_gaia_catalog_for_tess_file(
                tess_file=tess_fits_path,
                output_path=args.gaia_catalog_dir,
                pixel_padding=args.gaia_pixel_padding,
                magnitude_limit=args.gaia_magnitude_limit,
                force_download=args.force_gaia_download,
                gaia_credentials_file=args.gaia_credentials_file,
            )
        except Exception as e:
            print(f"❌ ERROR: Failed to download Gaia catalog: {e}")
            exit(1)

    # Call the processing function with parsed arguments
    results = process_tess_image_optimized(
        tess_file=tess_fits_path,
        skycell_wcs_csv=args.skycell_wcs_csv,
        output_path=args.output_path,
        pad_distance=args.pad_distance,
        edge_exclusion=args.edge_exclusion,
        edge_buffer_large=args.edge_buffer_large,
        edge_buffer_small=args.edge_buffer_small,
        buffer=args.buffer,
        tess_buffer=args.tess_buffer,
        n_threads=args.n_threads,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        oversampling_factor=args.oversampling_factor,
    )

    print(f"Processing results: {results}")
