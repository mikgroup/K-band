# Calculate SSIM, NMSE, and PSNR for reconstructed images.
# @author: Frederic Wang, Han Qi UC Berkeley, 2022

import h5py
import numpy as np
from utils_pipeline import ErrorMetrics, getPSNR

verbose = False

print("Start")
data_ori_img = h5py.File(
    "/mikRAID/han2019/brain_data_paper/brain_data_400samples_test.h5", "r"
)
data_ori_mask = h5py.File("/mikRAID/han2019/brain_data_paper/brain_2d_Rv4_test.h5", "r")
recon_images = np.load(
    "/mikRAID/han2019/brain_data_paper/brain_2d_Rv4_test.h5_L1unroll712_17.npy"
)

SSIM_array = []
NMSE_array = []
PSNR_array = []

for i in range(400):
    c = ErrorMetrics(data_ori_img["imgs"][i], recon_images[i])
    c.calc_SSIM()
    c.calc_NMSE()
    SSIM_array.append(c.SSIM)
    NMSE_array.append(c.NMSE)
    PSNR_array.append(getPSNR(abs(data_ori_img["imgs"][i]), abs(recon_images[i])))
    if verbose:
        print(
            f"Sample {i}: SSIM: {SSIM_array[-1]} NMSE: {NMSE_array[-1]} PSNR: {PSNR_array[-1]}"
        )

print("Statistics:")
print("NMSE")
# Convert to string to avoid floating point rounding errors.
print(
    f"{str(round(np.mean(NMSE_array), 6))[:8]} ± {str(round(np.std(NMSE_array), 6))[:8]}"
)

print("SSIM")
print(f"{round(np.mean(SSIM_array), 4)} ± {round(np.std(SSIM_array), 4)}")

print("PSNR")
print(f"{round(np.mean(PSNR_array), 4)} ± {round(np.std(PSNR_array), 4)}")
