import numpy as np


def compute_q_conj_prod(
    lap_pyr_1: np.ndarray,
    riesz_x_1: np.ndarray,
    riesz_y_1: np.ndarray,
    lap_pyr_2: np.ndarray,
    riesz_x_2: np.ndarray,
    riesz_y_2: np.ndarray,
):
    """Compute the product between the first frame Riesz pyramid and the conjugate
    of the second frame Riesz pyramid.

    The laplacian, riesz_x and riesz_y correspond, respectively, to the real, i and j
    components of the quaternion representing the pyramid.
    """
    q_conj_prod_real = (
        lap_pyr_1 * lap_pyr_2 + riesz_x_1 * riesz_x_2 + riesz_y_1 * riesz_y_2
    )
    q_conj_prod_x = -lap_pyr_1 * riesz_x_2 + lap_pyr_2 * riesz_x_1
    q_conj_prod_y = -lap_pyr_1 * riesz_y_2 + lap_pyr_2 * riesz_y_1

    return q_conj_prod_real, q_conj_prod_x, q_conj_prod_y


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    min_val: float = 1e-9,
    rtol: float = 1e-5,
):
    """Safe division of two arrays. A value of 0 is used where the division fails."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=~np.isclose(denominator, 0, rtol=rtol, atol=min_val),
    )
