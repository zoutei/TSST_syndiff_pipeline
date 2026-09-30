# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def evaluate_sip_polynomial(u, v, order, coeff_prefix, header):
    """
    Evaluate SIP polynomial correction
    
    Parameters:
    -----------
    u, v : float
        Pixel coordinates relative to reference pixel
    order : int
        Maximum polynomial order
    coeff_prefix : str
        'A' for f(u,v) or 'B' for g(u,v)
    header : dict
        FITS header containing SIP coefficients
        
    Returns:
    --------
    float : Polynomial correction value
    """
    correction = 0.0
    for i in range(order + 1):
        for j in range(order - i + 1):
            key = f"{coeff_prefix}_{i}_{j}"
            if key in header:
                term = header[key] * (u**i) * (v**j)
                correction += term
    return correction

def evaluate_inverse_sip_polynomial(u_prime, v_prime, order, coeff_prefix, header):
    """
    Evaluate inverse SIP polynomial correction
    
    This function applies the inverse distortion correction using AP or BP coefficients.
    These coefficients are pre-computed to directly map from distorted to undistorted coordinates.
    
    Parameters:
    -----------
    u_prime, v_prime : float
        Distortion-corrected pixel coordinates relative to reference pixel
    order : int
        Maximum polynomial order
    coeff_prefix : str
        'AP' for inverse f(u,v) or 'BP' for inverse g(u,v)
    header : dict
        FITS header containing inverse SIP coefficients
        
    Returns:
    --------
    float : Inverse polynomial correction value
    """
    correction = 0.0
    for i in range(order + 1):
        for j in range(order - i + 1):
            key = f"{coeff_prefix}_{i}_{j}"
            if key in header:
                term = header[key] * (u_prime**i) * (v_prime**j)
                correction += term
    return correction

def invert_sip_distortion_iterative(u_prime, v_prime, header, max_iter=50, tolerance=1e-10, use_ap_bp_guess=True):
    """
    Iteratively solve for undistorted coordinates from distorted ones using Newton-Raphson
    
    This function solves the system:
        u_prime = u + f(u,v)
        v_prime = v + g(u,v)
    for (u, v) given (u_prime, v_prime)
    
    Parameters:
    -----------
    u_prime, v_prime : float
        Distorted pixel coordinates relative to reference pixel
    header : dict
        FITS header containing forward SIP coefficients (A_, B_)
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    use_ap_bp_guess : bool
        If True and AP/BP coefficients are available, use them as initial guess
        
    Returns:
    --------
    tuple : (u, v) undistorted coordinates
    """
    # Initial guess: use AP/BP if available, otherwise assume undistorted = distorted
    if use_ap_bp_guess and 'AP_ORDER' in header and 'BP_ORDER' in header:
        # Use AP/BP coefficients for better initial guess
        ap_order = int(header['AP_ORDER'])
        bp_order = int(header['BP_ORDER'])
        
        f_inv = evaluate_inverse_sip_polynomial(u_prime, v_prime, ap_order, 'AP', header)
        g_inv = evaluate_inverse_sip_polynomial(u_prime, v_prime, bp_order, 'BP', header)
        
        u = u_prime + f_inv
        v = v_prime + g_inv
    else:
        # Simple initial guess
        u = u_prime
        v = v_prime
    
    a_order = int(header['A_ORDER'])
    b_order = int(header['B_ORDER'])
    
    for iteration in range(max_iter):
        # Compute forward distortion at current guess
        f_uv = evaluate_sip_polynomial(u, v, a_order, 'A', header)
        g_uv = evaluate_sip_polynomial(u, v, b_order, 'B', header)
        
        # Compute residuals
        residual_u = u + f_uv - u_prime
        residual_v = v + g_uv - v_prime
        
        # Check convergence
        if np.abs(residual_u) < tolerance and np.abs(residual_v) < tolerance:
            return u, v
        
        # Compute Jacobian (derivatives of distortion)
        # df/du, df/dv, dg/du, dg/dv
        h = 1e-6  # Small step for numerical derivatives
        
        f_u_plus = evaluate_sip_polynomial(u + h, v, a_order, 'A', header)
        f_v_plus = evaluate_sip_polynomial(u, v + h, a_order, 'A', header)
        g_u_plus = evaluate_sip_polynomial(u + h, v, b_order, 'B', header)
        g_v_plus = evaluate_sip_polynomial(u, v + h, b_order, 'B', header)
        
        df_du = (f_u_plus - f_uv) / h
        df_dv = (f_v_plus - f_uv) / h
        dg_du = (g_u_plus - g_uv) / h
        dg_dv = (g_v_plus - g_uv) / h
        
        # Jacobian matrix (I + J_distortion)
        J = np.array([[1 + df_du, df_dv], 
                      [dg_du, 1 + dg_dv]])
        
        # Solve for correction: J * delta = -residual
        residual = np.array([residual_u, residual_v])
        try:
            delta = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError:
            # If singular, just use the residual directly
            delta = -residual
        
        # Update guess
        u += delta[0]
        v += delta[1]
    
    # If we didn't converge, return best guess
    print(f"Warning: SIP inversion did not converge after {max_iter} iterations")
    return u, v

def forward_tan_projection(ra, dec, header):
    """
    Convert celestial coordinates to intermediate world coordinates using TAN projection
    
    Parameters:
    -----------
    ra, dec : float
        Right ascension and declination in degrees
    header : dict
        FITS header containing CRVAL1, CRVAL2
        
    Returns:
    --------
    tuple : (xi, eta) intermediate world coordinates in degrees
    """
    # Convert to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    
    # Reference coordinates
    ra0 = np.deg2rad(header["CRVAL1"])
    dec0 = np.deg2rad(header["CRVAL2"])
    
    # Handle special case at reference point
    if np.allclose(ra, header["CRVAL1"]) and np.allclose(dec, header["CRVAL2"]):
        return (0.0, 0.0)
    
    # Forward TAN projection
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    
    dra = ra_rad - ra0
    cos_dra = np.cos(dra)
    sin_dra = np.sin(dra)
    
    # Calculate angular distance
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    
    if cos_c <= 0:
        raise ValueError("Point is more than 90 degrees from reference point")
    
    # Intermediate coordinates
    xi = cos_dec * sin_dra / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / cos_c
    
    return (np.rad2deg(xi), np.rad2deg(eta))

def apply_sip_correction(x, y, header):
    """
    Apply SIP distortion correction to arrays of pixel coordinates.

    Parameters
    ----------
    x, y : np.ndarray
        Pixel coordinates (0-based).
    header : dict or fits.Header
        FITS header with SIP coefficients.

    Returns
    -------
    u_prime, v_prime : np.ndarray
        SIP-corrected pixel coordinates relative to reference pixel.
    """
    # Convert to detector coordinates (1-based FITS convention)
    u = x - (header['CRPIX1'] - 1)
    v = y - (header['CRPIX2'] - 1)
    a_order = int(header['A_ORDER'])
    b_order = int(header['B_ORDER'])

    # Vectorized SIP polynomial evaluation
    def eval_sip_poly(u, v, order, prefix):
        result = np.zeros_like(u, dtype=float)
        for i in range(order + 1):
            for j in range(order - i + 1):
                key = f"{prefix}_{i}_{j}"
                if key in header:
                    result += header[key] * (u ** i) * (v ** j)
        return result

    f_uv = eval_sip_poly(u, v, a_order, 'A')
    g_uv = eval_sip_poly(u, v, b_order, 'B')
    u_prime = u + f_uv
    v_prime = v + g_uv
    return u_prime, v_prime

def uvprime_to_radec(u_prime, v_prime, header):
    """
    Convert SIP-corrected pixel coordinates to RA/DEC.

    Parameters
    ----------
    u_prime, v_prime : np.ndarray
        SIP-corrected pixel coordinates relative to reference pixel.
    header : dict or fits.Header
        FITS header with WCS info.

    Returns
    -------
    ra, dec : np.ndarray
        RA and DEC in degrees.
    """
    # Linear transformation (CD matrix)
    xi = header['CD1_1'] * u_prime + header['CD1_2'] * v_prime
    eta = header['CD2_1'] * u_prime + header['CD2_2'] * v_prime

    # TAN projection (vectorized)
    xi_rad = np.deg2rad(xi)
    eta_rad = np.deg2rad(eta)
    ra0 = np.deg2rad(header["CRVAL1"])
    dec0 = np.deg2rad(header["CRVAL2"])

    rho = np.sqrt(xi_rad ** 2 + eta_rad ** 2)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    # Avoid division by zero at reference point
    with np.errstate(invalid='ignore', divide='ignore'):
        dec = np.arcsin(cos_c * sin_dec0 + (eta_rad * sin_c * cos_dec0) / np.where(rho == 0, 1, rho))
        y_term = xi_rad * sin_c
        x_term = rho * cos_dec0 * cos_c - eta_rad * sin_dec0 * sin_c
        ra = ra0 + np.arctan2(y_term, x_term)
        # At reference point, set to reference values
        dec = np.where(rho == 0, dec0, dec)
        ra = np.where(rho == 0, ra0, ra)

    return np.rad2deg(ra), np.rad2deg(dec)


def radec_to_uvprime(ra, dec, header):
    """
    Convert arrays of RA/DEC to SIP-corrected pixel coordinates (u', v') relative to reference pixel.

    Parameters
    ----------
    ra, dec : np.ndarray
        RA and DEC in degrees.
    header : dict or fits.Header
        FITS header with WCS info.

    Returns
    -------
    u_prime, v_prime : np.ndarray
        SIP-corrected pixel coordinates relative to reference pixel.
    """
    # Forward TAN projection (RA/DEC -> xi, eta)
    xi, eta = np.vectorize(lambda r, d: forward_tan_projection(r, d, header))(ra, dec)

    # Inverse CD matrix
    cd11 = header['CD1_1']
    cd12 = header['CD1_2']
    cd21 = header['CD2_1']
    cd22 = header['CD2_2']
    det = cd11 * cd22 - cd12 * cd21
    cd_inv = np.array([[cd22/det, -cd12/det],
                       [-cd21/det, cd11/det]])

    xi = np.array(xi)
    eta = np.array(eta)
    u_prime = cd_inv[0, 0] * xi + cd_inv[0, 1] * eta
    v_prime = cd_inv[1, 0] * xi + cd_inv[1, 1] * eta
    return u_prime, v_prime


def corrected_uv_to_xy(u_prime, v_prime, header):
    """
    Convert SIP-corrected pixel coordinates (u', v') to pixel coordinates (x, y) (0-based), applying inverse SIP distortion.

    Parameters
    ----------
    u_prime, v_prime : np.ndarray
        SIP-corrected pixel coordinates relative to reference pixel.
    header : dict or fits.Header
        FITS header with WCS info.

    Returns
    -------
    x, y : np.ndarray
        Pixel coordinates (0-based).
    """
    # Vectorized SIP inversion using the iterative method
    def invert_one(u_p, v_p):
        u, v = invert_sip_distortion_iterative(u_p, v_p, header, use_ap_bp_guess=True)
        return u, v
    
    u_arr = np.empty_like(u_prime)
    v_arr = np.empty_like(v_prime)
    for i in range(len(u_prime)):
        u_arr[i], v_arr[i] = invert_one(u_prime[i], v_prime[i])
    
    x = u_arr + (header['CRPIX1'] - 1)
    y = v_arr + (header['CRPIX2'] - 1)
    return x, y



if __name__ == "__main__":
    im1_file = "../data/tess_ffi/s0020_3_3/tess2019359015923-s0020-3-3-0165-s_ffic.fits"
    hdul = fits.open(im1_file)
    tess_header = hdul[1].header
    tess_wcs = WCS(hdul[1].header)
    hdul.close()


    # Test with sample x and y arrays
    x_samples = np.array([0, 100, 500, 1000, 1500])
    y_samples = np.array([0, 200, 600, 1200, 1800])

    u_prime, v_prime = apply_sip_correction(x_samples, y_samples, tess_header)
    ra_custom, dec_custom = corrected_to_radec(u_prime, v_prime, tess_header)

    # Astropy comparison
    astropy_ra, astropy_dec = tess_wcs.all_pix2world(x_samples, y_samples, 0)

    # Print comparison
    for i in range(len(x_samples)):
        print(f"Pixel ({x_samples[i]}, {y_samples[i]})")
        print(f"  Custom:   RA={ra_custom[i]:.6f}, DEC={dec_custom[i]:.6f}")
        print(f"  Astropy:  RA={astropy_ra[i]:.6f}, DEC={astropy_dec[i]:.6f}")
        print(f"  ΔRA={abs(ra_custom[i] - astropy_ra[i]):.2e}, ΔDEC={abs(dec_custom[i] - astropy_dec[i]):.2e}\n")


    # Quick test using ra_custom and dec_custom
    u_prime_test, v_prime_test = radec_to_corrected_uv(ra_custom, dec_custom, tess_header)
    x_test, y_test = corrected_uv_to_xy(u_prime_test, v_prime_test, tess_header)

    # Astropy comparison
    astropy_x, astropy_y = tess_wcs.all_world2pix(ra_custom, dec_custom, 0)

    for i in range(len(ra_custom)):
        print(f"RA/DEC ({ra_custom[i]:.6f}, {dec_custom[i]:.6f})")
        print(f"  Custom:   x={x_test[i]:.6f}, y={y_test[i]:.6f}")
        print(f"  Astropy:  x={astropy_x[i]:.6f}, y={astropy_y[i]:.6f}")
        print(f"  Δx={abs(x_test[i] - astropy_x[i]):.2e}, Δy={abs(y_test[i] - astropy_y[i]):.2e}\n")
