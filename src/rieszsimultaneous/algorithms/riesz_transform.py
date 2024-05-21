import cv2
import numpy as np


from typing import Sequence, Tuple


def compute_ideal_riesz_transform(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ideal Riesz transform of a 2D array.

    Parameters
    ----------
    im : np.ndarray
        Image to compute the Riesz transform.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Riesz transform in x and y directions.
    """
    if len(im.shape) != 2:
        raise ValueError(f"Im must be a 2-d numpy array, but has shape {im.shape}")

    # guarantee symmetric axes around zero
    h, w = im.shape
    h_padded = h if h % 2 != 0 else h + 1
    w_padded = w if w % 2 != 0 else w + 1

    # frequency space
    fx = np.arange(-(w_padded - 1) // 2, w_padded // 2 + 1) / w_padded
    fy = np.arange(-(h_padded - 1) // 2, h_padded // 2 + 1) / h_padded
    mfx, mfy = np.meshgrid(fx, fy)
    r = np.sqrt(mfx**2 + mfy**2)

    # frequency responses of riesz transform
    mask = ~np.isclose(r, 0, rtol=1e-7, atol=1e-11)
    rx = -1j * np.divide(mfx, r, out=np.ones_like(mfx), where=mask)
    ry = -1j * np.divide(mfy, r, out=np.ones_like(mfx), where=mask)

    # compute transform of im
    im_fft = np.fft.fftshift(np.fft.fft2(im, s=[h_padded, w_padded]))
    im_rx_fft = im_fft * rx
    im_ry_fft = im_fft * ry

    # return to spatial domain
    im_rx = np.fft.ifft2(np.fft.ifftshift(im_rx_fft))
    im_ry = np.fft.ifft2(np.fft.ifftshift(im_ry_fft))
    im_rx = np.real(im_rx[:h, :w])  # remove extra dimension used for symmetry
    im_ry = np.real(im_ry[:h, :w])  # remove extra dimension used for symmetry

    return im_rx, im_ry


def compute_approximate_riesz_transform(
    im: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute approximate Riesz transform of a 2D array.

    Parameters
    ----------
    im : np.ndarray
        Image to compute the Riesz transform.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Riesz transform in x and y directions.
    """
    kernel_x, kernel_y = get_approximate_riesz_transform_kernels()
    im_rx = cv2.filter2D(im, -1, kernel_x)
    im_ry = cv2.filter2D(im, -1, kernel_y)
    return im_rx, im_ry


def get_approximate_riesz_transform_kernels():
    kernel_x = np.zeros((3, 3), dtype=np.float32)
    kernel_x[1, 0] = 0.5
    kernel_x[1, 2] = -0.5

    kernel_y = np.zeros((3, 3), dtype=np.float32)
    kernel_y[0, 1] = 0.5
    kernel_y[2, 1] = -0.5
    return kernel_x, kernel_y


def compute_riesz_pyramid(
    im: np.ndarray,
    levels: int,
    riesz_transform_func: callable = compute_ideal_riesz_transform,
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
    """Compute the Riesz pyramid of a 2D array.

    Index 0 is the base of the pyramid and  the Real part has one more element
    which corresponds to the residue.

    Parameters
    ----------
    im : np.ndarray
        Image to compute the Riesz pyramid.
    levels : int
        Number of levels of the pyramid.

    Returns
    -------
    Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]
        List where each element corresponds to a level of the pyramid.
        Each element is a Tuple of laplacian, riesz_x and riesz_y.
    """
    lap_pyr = [np.zeros(im.shape, dtype=np.float32)]
    np.copyto(lap_pyr[0], im)

    riesz_x_pyr = []
    riesz_y_pyr = []
    for k in range(0, levels):
        # construct level k by computing its reduced version, expanding it and subtracting it
        lap_down = cv2.pyrDown(lap_pyr[k])
        lap_pyr.append(lap_down)
        lap_up = cv2.pyrUp(lap_down, dstsize=lap_pyr[k].shape[1::-1])
        lap_pyr[k] -= lap_up

        im_riesz_x, im_riesz_y = riesz_transform_func(lap_pyr[k])

        riesz_x_pyr.append(im_riesz_x)
        riesz_y_pyr.append(im_riesz_y)

    return lap_pyr, riesz_x_pyr, riesz_y_pyr


def compute_ideal_riesz_pyramid(
    im: np.ndarray, levels: int
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
    return compute_riesz_pyramid(im, levels, compute_ideal_riesz_transform)


def compute_approximate_riesz_pyramid(
    im: np.ndarray, levels: int
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
    return compute_riesz_pyramid(im, levels, compute_approximate_riesz_transform)
