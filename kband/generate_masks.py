# Generate sampling + loss masks for the network, with customizable type and undersampling method
# @author: Frederic Wang, UC Berkeley, 2022

from optparse import OptionParser

import h5py
import numpy as np
import ssdu_masks as ssdu_masks
from utils_pipeline import band_mask, square_mask, vardens_mask, vardens_mask_1d


def gen_masks(config, mask_type, undersample, one_dim=False):
    """Generates sampling masks and loss masks for the network.
    Loss masks simulated undersampled training data, and are used to calculate training loss. Therefore they are different with respective to different experiments.
    For fully supervised training - The loss mask has 1s everywhere.
    For k-band - The loss mask contains 1s within the band and 0s outside. The angle of the band-mask is also varied.
    For SSDU - Consistent with the SSDU paper, the loss mask is a disjoint set of the original mask by Gaussian splitting. Please refer to the details in ssdu_masks.py (Yaman et al., 2020, MRM).
    For the vertical baseline - The loss mask is a fixed vertical band in the center, containing all 1s within the band and 0s outside.
    For the square baseline - The loss mask is a fixed square area in the center of k-space, containing all 1s within the band and 0s outside.
    Args:
        config (dict): mask parameters. 
            -config['H']: training/validation height
            -config['W']: training/validation width
            -config['inference_H']: inference height
            -config['inference_W']: inference width
            -config['R']: inference undersampling rate
            -config['R_band']: training unsampling rate
            -config['calib']: calibration area
        mask_type (string): can be supervised, kband, SSDU, vertical, or square
        undersample (function): undersampling function, for 2d undersampling it is mr.poisson
        one_dim (boolean): if undersampling is 1d
    Returns:
        sampling_masks, loss_masks (2-dimensional ndarrays)"""

    undersample_mask = undersample(
        config["H"], config["W"], config["R"], config["calib"]
    ).astype(np.float64)
    if mask_type == "supervised":
        sampling_mask = undersample_mask
        loss_mask = np.ones((config["H"], config["W"]))
    elif mask_type == "kband":
        blade_angle = np.random.randint(0, 180)
        loss_mask = band_mask(config["H"], config["W"], blade_angle, config["R_band"])
        sampling_mask = undersample_mask * loss_mask
    elif mask_type == "SSDU":
        if one_dim:
            sampling_mask, loss_mask = ssdu_masker.Gaussian_selection(
                undersample_mask[0:1]
            )
            sampling_mask = np.tile(sampling_mask, (config["H"], 1))
            loss_mask = np.tile(loss_mask, (config["H"], 1))
        else:
            sampling_mask, loss_mask = ssdu_masker.Gaussian_selection(undersample_mask)
    elif mask_type == "vertical":
        loss_mask = band_mask(config["H"], config["W"], 0, config["R_band"])
        sampling_mask = undersample_mask * loss_mask
    elif mask_type == "square":
        loss_mask = square_mask(config["H"], config["W"], config["R_band"])
        sampling_mask = undersample_mask * loss_mask
    return sampling_mask, loss_mask


def get_args():
    parser = OptionParser()
    parser.add_option(
        "-c",
        "--config",
        dest="config",
        default="knee",
        help="Which default config to create masks with, knee or brain or other.",
    )
    parser.add_option(
        "-t",
        "--type",
        dest="type",
        default="SSDU",
        help="Which masking type: kband, SSDU, square, vertical, or supervised",
    )
    parser.add_option(
        "-u",
        "--undersampling",
        dest="undersampling",
        default="1d",
        help="Which undersampling method: 2d, 1d, or radial.",
    )
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    # Knee and Brain data configs from FastMRI
    fastMRI_knee_config = {
        "output": "/mikRAID/fredwang/knee_data/",
        "n_t_data": 1600,
        "n_v_data": 400,
        "n_i_data": 400,
        "R": 4,
        "R_band": 4,
        "H": 400,
        "W": 300,
        "inference_H": 640,
        "inference_W": 372,
        "calib": 20,
    }
    fastMRI_brain_config = {
        "output": "/mikRAID/han2019/brain_data_paper/",
        "n_t_data": 1600,
        "n_v_data": 400,
        "n_i_data": 400,
        "R": 4,
        "R_band": 4,
        "H": 320,
        "W": 230,
        "inference_H": 640,
        "inference_W": 320,
        "calib": 21,
    }
    # Empty template.
    other_config = {
        "n_t_data": 0,
        "n_v_data": 0,
        "n_i_data": 0,
        "R": 0,
        "R_band": 0,
        "H": 0,
        "W": 0,
        "inference_H": 0,
        "inference_W": 0,
        "calib": 0,
    }

    args = get_args()
    if args.config == "knee":
        config = fastMRI_knee_config
    elif args.config == "brain":
        config = fastMRI_brain_config
    elif args.config == "other":
        config = other_config
    else:
        raise ValueError(
            "Unknown config type specified. Must be 'knee' or 'brain' or 'other'"
        )

    if args.undersampling == "2d":
        undersample = vardens_mask
    elif args.undersampling == "1d":
        undersample = vardens_mask_1d
    elif args.undersampling == "radial":
        raise Exception("Not implemented yet.")

    # Training masks.
    train_sampling_masks = np.zeros(
        (config["n_t_data"], config["H"], config["W"]), dtype=np.float64
    )
    train_loss_masks = np.zeros(
        (config["n_t_data"], config["H"], config["W"]), dtype=np.float64
    )
    ssdu_masker = ssdu_masks.ssdu_masks()

    for i in range(config["n_t_data"]):
        train_sampling_masks[i], train_loss_masks[i] = gen_masks(
            config, args.type, undersample, args.undersampling == "1d"
        )
        print(i + 1, "out of", config["n_t_data"], "training masks done")

    f = h5py.File(
        "{}{}_{}{}_Rv{}_Rb{}_train.h5".format(
            config["output"],
            args.config,
            args.type,
            args.undersampling,
            config["R"],
            config["R_band"],
        ),
        "w",
    )
    f.create_dataset("masks", data=abs(train_sampling_masks), dtype=np.float64)
    f.create_dataset("loss_masks", data=abs(train_loss_masks), dtype=np.float64)
    f.close()

    # Validation masks.
    val_sampling_masks = np.zeros(
        (config["n_v_data"], config["H"], config["W"]), dtype=np.float64
    )
    val_loss_masks = np.zeros(
        (config["n_v_data"], config["H"], config["W"]), dtype=np.float64
    )

    for i in range(config["n_v_data"]):
        val_sampling_masks[i], val_loss_masks[i] = gen_masks(
            config, args.type, undersample, args.undersampling == "1d"
        )
        print(i + 1, "out of", config["n_v_data"], "validation masks done")

    f = h5py.File(
        "{}{}_{}{}_Rv{}_Rb{}_val.h5".format(
            config["output"],
            args.config,
            args.type,
            args.undersampling,
            config["R"],
            config["R_band"],
        ),
        "w",
    )
    f.create_dataset("masks", data=abs(val_sampling_masks), dtype=np.float64)
    f.create_dataset("loss_masks", data=abs(val_loss_masks), dtype=np.float64)
    f.close()

    # Inference masks.
    test_sampling_masks = np.zeros(
        (config["n_i_data"], config["inference_H"], config["inference_W"]),
        dtype=np.float64,
    )
    test_loss_masks = np.zeros(
        (config["n_i_data"], config["inference_H"], config["inference_W"]),
        dtype=np.float64,
    )

    for i in range(config["n_i_data"]):
        test_sampling_masks[i] = undersample(
            config["inference_H"], config["inference_W"], config["R"], config["calib"]
        ).astype(np.float64)
        test_loss_masks[i] = np.ones((config["inference_H"], config["inference_W"]))
        print(i + 1, "out of", config["n_i_data"], "inference masks done")

    f = h5py.File(
        "{}{}_{}_Rv{}_test.h5".format(
            config["output"], args.config, args.undersampling, config["R"]
        ),
        "w",
    )
    f.create_dataset("masks", data=abs(test_sampling_masks), dtype=np.float64)
    f.create_dataset("loss_masks", data=abs(test_loss_masks), dtype=np.float64)
    f.close()
