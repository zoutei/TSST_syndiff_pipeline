"""SIP-aware WCS coordinate conversions (TESS / FITS headers)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def cd_matrix_from_header(header: fits.Header | dict[str, Any]) -> np.ndarray:
    """Return the 2x2 CD matrix from CD or PC+CDELT keywords."""
    hdr = fits.Header(header) if not isinstance(header, fits.Header) else header
    if "CD1_1" in hdr:
        return np.array(
            [
                [float(hdr["CD1_1"]), float(hdr["CD1_2"])],
                [float(hdr["CD2_1"]), float(hdr["CD2_2"])],
            ],
            dtype=float,
        )
    if all(k in hdr for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2")):
        pc = np.array(
            [
                [float(hdr["PC1_1"]), float(hdr["PC1_2"])],
                [float(hdr["PC2_1"]), float(hdr["PC2_2"])],
            ],
            dtype=float,
        )
        cdelt = np.array([float(hdr["CDELT1"]), float(hdr["CDELT2"])], dtype=float)
        return pc * cdelt[:, np.newaxis]
    return np.array(WCS(hdr).pixel_scale_matrix, dtype=float)


def ensure_cd_header(
    header: fits.Header | dict[str, Any],
    *,
    inplace: bool = False,
) -> fits.Header:
    """Write CD1_1..CD2_2 and remove PC/CDELT when missing."""
    hdr = header if inplace and isinstance(header, fits.Header) else deepcopy(fits.Header(header))
    if "CD1_1" not in hdr:
        cd = cd_matrix_from_header(hdr)
        hdr["CD1_1"] = float(cd[0, 0])
        hdr["CD1_2"] = float(cd[0, 1])
        hdr["CD2_1"] = float(cd[1, 0])
        hdr["CD2_2"] = float(cd[1, 1])
    for key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
        if key in hdr:
            del hdr[key]
    return hdr


def _header_with_cd(header: fits.Header | dict[str, Any]) -> fits.Header:
    return ensure_cd_header(header, inplace=False)


def evaluate_sip_polynomial(u, v, order, coeff_prefix, header):
    correction = 0.0
    for i in range(order + 1):
        for j in range(order - i + 1):
            key = f"{coeff_prefix}_{i}_{j}"
            if key in header:
                correction += header[key] * (u**i) * (v**j)
    return correction


def evaluate_inverse_sip_polynomial(u_prime, v_prime, order, coeff_prefix, header):
    correction = 0.0
    for i in range(order + 1):
        for j in range(order - i + 1):
            key = f"{coeff_prefix}_{i}_{j}"
            if key in header:
                correction += header[key] * (u_prime**i) * (v_prime**j)
    return correction


def invert_sip_distortion_iterative(
    u_prime,
    v_prime,
    header,
    max_iter=50,
    tolerance=1e-10,
    use_ap_bp_guess=True,
):
    if use_ap_bp_guess and "AP_ORDER" in header and "BP_ORDER" in header:
        ap_order = int(header["AP_ORDER"])
        bp_order = int(header["BP_ORDER"])
        f_inv = evaluate_inverse_sip_polynomial(u_prime, v_prime, ap_order, "AP", header)
        g_inv = evaluate_inverse_sip_polynomial(u_prime, v_prime, bp_order, "BP", header)
        u = u_prime + f_inv
        v = v_prime + g_inv
    else:
        u = u_prime
        v = v_prime

    a_order = int(header["A_ORDER"])
    b_order = int(header["B_ORDER"])

    for _ in range(max_iter):
        f_uv = evaluate_sip_polynomial(u, v, a_order, "A", header)
        g_uv = evaluate_sip_polynomial(u, v, b_order, "B", header)
        residual_u = u + f_uv - u_prime
        residual_v = v + g_uv - v_prime
        if np.abs(residual_u) < tolerance and np.abs(residual_v) < tolerance:
            return u, v
        h = 1e-6
        f_u_plus = evaluate_sip_polynomial(u + h, v, a_order, "A", header)
        f_v_plus = evaluate_sip_polynomial(u, v + h, a_order, "A", header)
        g_u_plus = evaluate_sip_polynomial(u + h, v, b_order, "B", header)
        g_v_plus = evaluate_sip_polynomial(u, v + h, b_order, "B", header)
        df_du = (f_u_plus - f_uv) / h
        df_dv = (f_v_plus - f_uv) / h
        dg_du = (g_u_plus - g_uv) / h
        dg_dv = (g_v_plus - g_uv) / h
        jacobian = np.array([[1 + df_du, df_dv], [dg_du, 1 + dg_dv]])
        residual = np.array([residual_u, residual_v])
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            delta = -residual
        u += delta[0]
        v += delta[1]
    return u, v


def forward_tan_projection(ra, dec, header):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    ra0 = np.deg2rad(header["CRVAL1"])
    dec0 = np.deg2rad(header["CRVAL2"])
    if np.allclose(ra, header["CRVAL1"]) and np.allclose(dec, header["CRVAL2"]):
        return (0.0, 0.0)
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    dra = ra_rad - ra0
    cos_dra = np.cos(dra)
    sin_dra = np.sin(dra)
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    if cos_c <= 0:
        raise ValueError("Point is more than 90 degrees from reference point")
    xi = cos_dec * sin_dra / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / cos_c
    return (np.rad2deg(xi), np.rad2deg(eta))


def apply_sip_correction(x, y, header):
    hdr = _header_with_cd(header)
    u = x - (hdr["CRPIX1"] - 1)
    v = y - (hdr["CRPIX2"] - 1)
    a_order = int(hdr["A_ORDER"])
    b_order = int(hdr["B_ORDER"])

    def eval_sip_poly(u_arr, v_arr, order, prefix):
        result = np.zeros_like(u_arr, dtype=float)
        for i in range(order + 1):
            for j in range(order - i + 1):
                key = f"{prefix}_{i}_{j}"
                if key in hdr:
                    result += hdr[key] * (u_arr**i) * (v_arr**j)
        return result

    f_uv = eval_sip_poly(u, v, a_order, "A")
    g_uv = eval_sip_poly(u, v, b_order, "B")
    return u + f_uv, v + g_uv


def uvprime_to_radec(u_prime, v_prime, header):
    hdr = _header_with_cd(header)
    cd = cd_matrix_from_header(hdr)
    xi = cd[0, 0] * u_prime + cd[0, 1] * v_prime
    eta = cd[1, 0] * u_prime + cd[1, 1] * v_prime
    xi_rad = np.deg2rad(xi)
    eta_rad = np.deg2rad(eta)
    ra0 = np.deg2rad(hdr["CRVAL1"])
    dec0 = np.deg2rad(hdr["CRVAL2"])
    rho = np.sqrt(xi_rad**2 + eta_rad**2)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(
            cos_c * sin_dec0 + (eta_rad * sin_c * cos_dec0) / np.where(rho == 0, 1, rho)
        )
        y_term = xi_rad * sin_c
        x_term = rho * cos_dec0 * cos_c - eta_rad * sin_dec0 * sin_c
        ra = ra0 + np.arctan2(y_term, x_term)
        dec = np.where(rho == 0, dec0, dec)
        ra = np.where(rho == 0, ra0, ra)
    return np.rad2deg(ra), np.rad2deg(dec)


def radec_to_uvprime(ra, dec, header):
    hdr = _header_with_cd(header)
    xi, eta = np.vectorize(lambda r, d: forward_tan_projection(r, d, hdr))(ra, dec)
    cd = cd_matrix_from_header(hdr)
    det = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    cd_inv = np.array([[cd[1, 1] / det, -cd[0, 1] / det], [-cd[1, 0] / det, cd[0, 0] / det]])
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    u_prime = cd_inv[0, 0] * xi + cd_inv[0, 1] * eta
    v_prime = cd_inv[1, 0] * xi + cd_inv[1, 1] * eta
    return u_prime, v_prime


def corrected_uv_to_xy(u_prime, v_prime, header):
    hdr = _header_with_cd(header)

    def invert_one(u_p, v_p):
        return invert_sip_distortion_iterative(u_p, v_p, hdr, use_ap_bp_guess=True)

    u_arr = np.empty_like(u_prime, dtype=float)
    v_arr = np.empty_like(v_prime, dtype=float)
    for i in range(len(u_prime)):
        u_arr[i], v_arr[i] = invert_one(u_prime[i], v_prime[i])
    x = u_arr + (hdr["CRPIX1"] - 1)
    y = v_arr + (hdr["CRPIX2"] - 1)
    return x, y


def radec_to_uv(ra, dec, header) -> tuple[np.ndarray, np.ndarray]:
    """World coordinates to detector offsets (u, v) relative to CRPIX."""
    hdr = _header_with_cd(header)
    u_p, v_p = radec_to_uvprime(ra, dec, hdr)
    x, y = corrected_uv_to_xy(u_p, v_p, hdr)
    crpix1 = float(hdr["CRPIX1"])
    crpix2 = float(hdr["CRPIX2"])
    return x - (crpix1 - 1.0), y - (crpix2 - 1.0)
