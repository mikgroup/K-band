# Utility functions for k-band pipeline.
# @author: Frederic Wang, Han Qi UC Berkeley, 2022

import matplotlib.pyplot as plt
import numpy as np
import sigpy.mri as mr
from PIL import Image
from SSIM_PIL import compare_ssim
from scipy.ndimage.interpolation import rotate


def square_mask(height=400, width=300, R_square_area=3):
    """Generates a numpy array square mask with specified angle and dimensions with area 1/R_square_area of the space.
    The sampling is done around the k-space center, thus sampling only low-resolution data.
    Args:
        height (int): number of rows in the band mask (default 400)
        width (int): number of columns in the band mask (default 300)
        R_square_area(double): R_band is the relative ratio that the band takes out of the full image size. For example, 
                        if an image is 100x100 pixels and R_band=4, then the square size will be 50x50. 
                        In our implementation, R_band must be higher than 1 and smaller than 10, i.e. the band can cover 
                        anything between 100% of kspace to 10% of kspace. We avoid very narrow bands since a 
                        fully sampled calibration area is required in kspace center for computing the sensitivity maps.
    Returns:
        mask (ndarray): Mask to be applied during loss.
    """
    assert R_square_area >= 1
    assert R_square_area <= 10

    num_pixels_in_square = height * width / R_square_area
    square_length = int(np.sqrt(num_pixels_in_square))
    mask = np.zeros((height, width))
    mask[
        int(height / 2 - square_length / 2) : int(height / 2 + square_length / 2),
        int(width / 2 - square_length / 2) : int(width / 2 + square_length / 2),
    ] = 1
    return mask


def band_mask(height=400, width=300, angle=45, R_band=3.0):
    """Generates a numpy array k-band mask with specified angle and dimensions with thickness 1/R_band of the space. 
    To sample all the data within the band, the band mask has 1s inside the band and 0s outside the band.
    The band also has an orientation that is defined by the input param angle. 
    The band height and width are computed such that the band will cover 1/R_band of the entire k-space, as described below.
    Args:
        height (int): number of rows in the band mask (default 400)
        width (int): number of columns in the band mask (default 300)
        angles (int): which band angle to generate the mask at.
        R_band(double): R_band is the relative ratio that the band takes out of the full image size. For example, 
                        if an image is 100x100 pixels and R_band=2, then the band size will be 50x100 or 100x50 
                        if the band axis aligns with the x or y directions (the band axis is defined by the angle). 
                        In our implementation, R_band must be higher than 1 and smaller than 10, i.e. the band can cover 
                        anything between 100% of kspace to 10% of kspace. We avoid very narrow bands since a 
                        fully sampled calibration area is required in kspace center for computing the sensitivity maps.
    Returns:
        mask (ndarray): Band supervision mask to be applied during loss.
    """
    assert R_band >= 1
    assert R_band <= 10
    assert angle >= 0
    assert angle <= 180

    dim = max(height, width)
    mask = np.zeros((2 * dim, 2 * dim))
    rotation_angle = angle * np.pi / 180
    angle = (180 - angle if angle > 90 else angle) * np.pi / 180

    # Band mask is tangent to top and bottom edges of k-space. Use the following equations:
    # mask_height * mask_width = height * width / R_band
    # mask_width * sin(angle) + mask_height * cos(angle) = height
    if angle < np.arctan2(width, height):
        mask_height = int(
            (
                height * R_band
                + np.sqrt(
                    (R_band * height) ** 2
                    - 4 * R_band * height * width * np.sin(angle) * np.cos(angle)
                )
            )
            / (2 * R_band * np.cos(angle))
        )
        mask_width = int(height * width / (R_band * mask_height))

    # Band mask is tangent to left and right edges of k-space. Use the following equations:
    # mask_height * mask_width = height * width / R_band
    # mask_width * cos(angle) + mask_height * sin(angle) = width
    else:
        mask_height = int(
            (
                width * R_band
                + np.sqrt(
                    (R_band * width) ** 2
                    - 4 * R_band * height * width * np.cos(angle) * np.sin(angle)
                )
            )
            / (2 * R_band * np.sin(angle))
        )
        mask_width = int(height * width / (R_band * mask_height))

    mask[
        dim - (mask_height // 2) : dim + (mask_height // 2),
        dim - (mask_width // 2) : dim + (mask_width // 2),
    ] += 1
    mask = rotate(mask, angle=rotation_angle * 180 / np.pi)
    side = mask.shape[0]

    return mask[
        int((side - height) / 2) : int((side - height) / 2) + height,
        int((side - width) / 2) : int((side - width) / 2) + width,
    ]


def vardens_mask(height=400, width=300, R=4, calib_size=20):
    """Creates a poisson disc mask of specified width, height, acceleration factor, and calibration area"""
    vd_mask = mr.poisson(
        (height, width), accel=R, crop_corner=False, seed=np.random.randint(0, 2 ** 32)
    )
    vd_mask[
        int(height / 2 - calib_size / 2) : int(height / 2 + calib_size / 2),
        int(width / 2 - calib_size / 2) : int(width / 2 + calib_size / 2),
    ] = 1
    return vd_mask


def vardens_mask_1d(height=400, width=300, R=4, calib_size=20):
    """Creates a poisson disc mask of specified width, height, acceleration factor, and calibration area"""
    vd_mask = mr.poisson(
        (height, width),
        accel=R * 1.8,
        crop_corner=False,
        seed=np.random.randint(0, 2 ** 32),
    )[height // 2]
    vd_mask[int(width / 2 - calib_size / 2) : int(width / 2 + calib_size / 2)] = 1
    return np.tile(vd_mask, (height, 1))


def imshowgray(im, vmin=None, vmax=None):
    """Displays image in grayscale"""
    im = np.flipud(im)
    plt.imshow(im, cmap=plt.get_cmap("gray"), vmin=vmin, vmax=vmax)


# @Author: Efrat Shimron
class ErrorMetrics:
    def __init__(self, I_true, I_pred):
        # convert images from complex to magnitude (we do not want complex data for error calculation)
        self.I_true = np.abs(I_true)
        self.I_pred = np.abs(I_pred)

    def calc_NMSE(self):
        # Reshape the images into vectors
        I_true = np.reshape(self.I_true, (1, -1))
        I_pred = np.reshape(self.I_pred, (1, -1))
        # Mean Square Error
        self.MSE = np.square(np.subtract(I_true, I_pred)).mean()
        rr = np.max(I_true) - np.min(I_true)  # range
        self.NMSE = self.MSE / rr

    def calc_SSIM(self):
        # Note: in order to use the function compare_ssim, the images must be converted to PIL format
        # convert the images from float32 to uint8 format
        im1_mag_uint8 = (self.I_true * 255 / np.max(self.I_true)).astype("uint8")
        im2_mag_uint8 = (self.I_pred * 255 / np.max(self.I_pred)).astype("uint8")
        # convert from numpy array to PIL format
        im1_PIL = Image.fromarray(im1_mag_uint8)
        im2_PIL = Image.fromarray(im2_mag_uint8)

        self.SSIM = compare_ssim(im1_PIL, im2_PIL)


def getSSIM(org, recon):
    """ Measures SSIM (Wang et al., 2004, IEEE Transactions on Image Processing) between the reference and the reconstructed images
    """
    c = error_metrics(org, recon)
    c.calc_SSIM()
    return c.SSIM


def getNMSE(org, recon):
    """ Measures NMSE between the reference and the reconstructed images 
    """
    c = error_metrics(org, recon)
    c.calc_NMSE()
    return c.NMSE


# Reference: TODO
def div0(a, b):
    """ This function handles division by zero """
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return c


def normalize01(img):
    """
    Normalize the image between 0 and 1
    """
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        img2[i] = div0(img[i] - img[i].min(), img[i].ptp())
    return np.squeeze(img2).astype(img.dtype)


def getPSNR(org, recon):
    """ Measures PSNR between the reference and the reconstructed images 
    """
    org = normalize01(org)
    recon = normalize01(recon)
    mse = np.sum(np.square(np.abs(org - recon))) / org.size
    psnr = 20 * np.log10(org.max() / (np.sqrt(mse) + 1e-10))
    return psnr


def fft2c(x):
    return (
        1
        / np.sqrt(np.prod(x.shape))
        * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    )


def ifft2c(y):
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(y))
    )
