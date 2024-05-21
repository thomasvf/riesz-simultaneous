import cv2
import numpy as np
import matplotlib.pyplot as plt


def rectangular_mask(shape, p0, p1):
    """
    Build rectangular mask with given parameters from p0 to p1.

    Parameters
    ----------
    shape : tuple
        Shape of the mask
    p0 : tuple
        Top left rectangle point in format (x, y)
    p1 : tuple
        Bottom right rectangle point in format (x, y)
    Returns
    -------
    mask : np.ndarray
        Mask whose values are 1 inside the rectangle and 0 outside
    """
    m = np.zeros(shape, dtype=np.float32)
    x0, y0 = p0
    x1, y1 = p1
    m[y0:y1, x0:x1] = 1.0
    return m


def chrominance_similarity_mask(im, p0, threshold, show_distances=False):
    """
    Compute a mask based on chrominance similarity between the given pixel and the rest of the image using the
    euclidean distance between the colors.

    Parameters
    ----------
    im : np.ndarray
        Reference frame for building the mask. Values should be in range [0, 255]
    p0 : tuple
        Reference pixel
    threshold : float
        Maximum difference between chrominance channels in p0 and in the rest of the image
    show_distances : bool
        If True, plot map of distances before closing

    Returns
    -------
    mask : np.ndarray
        Mask whose values inside the threshold are 1 and 0 outside
    """
    im_yiq = cv2.cvtColor(im.astype(np.float32) * 1.0 / 255, cv2.COLOR_RGB2YUV)
    x, y = p0

    chrominances = im_yiq[:, :, 1:]
    pix_chrominance = chrominances[y, x]
    distances = np.sqrt(
        (chrominances[:, :, 0] - pix_chrominance[0])**2 + (chrominances[:, :, 1] - pix_chrominance[1])**2)

    if show_distances:
        plt.imshow(distances)
        plt.show()

    mask = np.zeros_like(im)
    mask[:, :, 0] = 1.0*(distances < threshold)
    mask[:, :, 1] = 1.0*(distances < threshold)
    mask[:, :, 2] = 1.0*(distances < threshold)

    return mask
