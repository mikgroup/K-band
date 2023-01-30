# Visualize reconstructed image and calculate error metrics with respect to the ground truth.
# @author: Han Qi, Frederic Wang, UC Berkeley, 2022

import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils_pipeline import getPSNR, imshowgray

# Load the data.
data_file = ...  # ground-truth file
fully_sampled_data = h5py.File(data_file, "r")

supervised_recon = np.load(...)  # reconstruction with supervised training
kband_recon = np.load(...)  # reconstruction with k-band strategy
ssdu_recon = np.load(...)  # reconstruction with SSDU strategy

# Show zoomed in image

for i in range(supervised_recon.shape[0]):
    fig = plt.figure(figsize=(75, 53))
    vmax = (np.percentile(abs(fully_sampled_data["imgs"][i])[::-1], 100)) * 1.2
    vmin = np.percentile(abs(fully_sampled_data["imgs"][i])[::-1], 0)
    ax0 = fig.add_subplot(2, 4, 1)

    imshowgray(abs(fully_sampled_data["imgs"][i][::-1])[:, :], vmin=vmin, vmax=vmax)
    plt.axis("off")
    ax1 = fig.add_subplot(2, 4, 2)
    imshowgray(abs(supervised_recon[i][::-1])[:, :], vmin=vmin, vmax=vmax)
    plt.axis("off")
    psnr = getPSNR(fully_sampled_data["imgs"][i], supervised_recon[i])
    plt.text(
        10,
        20,
        "PSNR: {}".format(str(psnr)[1:6]),
        fontdict={"fontsize": 90, "color": "white", "weight": "bold"},
    )

    ax3 = fig.add_subplot(2, 4, 3)
    imshowgray(abs(ssdu_recon[i][::-1])[:, :], vmin=vmin, vmax=vmax)
    plt.axis("off")
    psnr = getPSNR(fully_sampled_data["imgs"][i], ssdu_recon[i])
    plt.text(
        10,
        20,
        "PSNR: {}".format(str(psnr)[1:6]),
        fontdict={"fontsize": 90, "color": "white", "weight": "bold"},
    )

    ax4 = fig.add_subplot(2, 4, 4)
    imshowgray(abs(kband_recon[i][::-1])[:, :], vmin=vmin, vmax=vmax)
    plt.axis("off")
    psnr = getPSNR(fully_sampled_data["imgs"][i], kband_recon[i])
    plt.text(
        10,
        20,
        "PSNR: {}".format(str(psnr)[1:6]),
        fontdict={"fontsize": 90, "color": "white", "weight": "bold"},
    )

    ax7 = fig.add_subplot(2, 4, 6)
    imshowgray(
        abs(supervised_recon[i][::-1] - fully_sampled_data["imgs"][i][::-1])[:, :],
        vmin=vmin,
        vmax=0.07 * vmax,
    )
    plt.axis("off")
    ax8 = fig.add_subplot(2, 4, 7)
    imshowgray(
        abs(ssdu_recon[i][::-1] - fully_sampled_data["imgs"][i][::-1])[:, :],
        vmin=vmin,
        vmax=0.07 * vmax,
    )
    plt.axis("off")
    ax8 = fig.add_subplot(2, 4, 8)
    imshowgray(
        abs(kband_recon[i][::-1] - fully_sampled_data["imgs"][i][::-1])[:, :],
        vmin=vmin,
        vmax=0.07 * vmax,
    )
    plt.axis("off")
    plt.subplots_adjust(wspace=0.00, hspace=0.00)
    ax0.set_title("Ground Truth", fontdict={"fontsize": 100, "weight": "bold"})
    ax1.set_title("MoDL", fontdict={"fontsize": 100, "weight": "bold"})
    ax3.set_title("SSDU", fontdict={"fontsize": 100, "weight": "bold"})
    ax4.set_title("K-Band", fontdict={"fontsize": 100, "weight": "bold"})
    plt.axis("off")
    fig.savefig(fname="visual_data_" + str(i), dpi=50)
