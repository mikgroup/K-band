# Generates sensitivity maps and the sensitivity weighted coil-combined target images. using raw h5 fastMRI data with BART.
# @author: Frederic Wang, Ke Wang, UC Berkeley, 2022

import os
from optparse import OptionParser

import bart
import h5py
import numpy as np
import sigpy as sp


def create_maps(input_dir, output_dir, config):
    """Generates sensitivity maps from data in input_dir and saves to output_dir
    Args:
        input_dir (string): path to processed data.
        output_dir (int): path to save data w/ sensitivity maps.
        config (dict): input generation config, reference line 132 for an example.
    """
    case = 0
    filenames = os.listdir(input_dir)
    for name in filenames:
        f = h5py.File(input_dir + "/" + name, "r")
        # Filter out which data files to use.
        if (
            f["kspace"].shape[1] == config["coils"]
            and f["kspace"].shape[2] == config["height"]
            and f["kspace"].shape[3] == config["width"]
            and f.attrs["acquisition"] in config["label"]
        ):
            print("Preparing ", name)

            # Keep the middle config['slices_keep'] slices, to maximize structure.
            middle_slice = len(f["kspace"]) // 2
            kspace = np.array(f["kspace"])[
                middle_slice
                - (config["slices_keep"] // 2) : middle_slice
                + (config["slices_keep"] // 2)
            ]
            case += 1

            # Normalize k-space to 95th percentile.
            # The 95th percentile is computed from magnitude low-res images, in the image domain (not k-space data). However, the normalization is done in k-space, since it doesn't matter if we multiply the data by a factor in image space or in k-space.
            # The aim of the normalization is to guarantee that the images will have intensity values around the range of [0,1], as neural networks are best adapted to this range.
            image = sp.rss(sp.ifft(kspace, axes=(-1, -2)), axes=1)
            scale = np.percentile(image.reshape(-1), 95)
            kspace_norm = kspace / scale

            for sli in range(config["slices_keep"]):
                kspace_slice = kspace_norm[sli, ...]  # save ksp

                # Generate sensitivity maps for each coil, with calibration region 20 or 21. We only compute the first set of ESPIRIT maps using the m1 flag.
                try:
                    sens = bart.bart(
                        1,
                        "ecalib -r {} -m1".format(config["calib"]),
                        kspace_slice.transpose(1, 2, 0)[None, ...],
                    )[0, ...].transpose(2, 0, 1)

                    # Here we compute the target, which are the sensitivity-weighted coil-combined complex-valued images.
                    # These images are computed based on the fully sampled k-space data.
                    # Notice that we compute them by running BART pics with only 1 iteration - this performs the sensitivity-weighted coil combination.
                    im_coil_combined = bart.bart(
                        1,
                        "pics -i 1 -S",
                        kspace_slice.transpose(1, 2, 0)[..., None, :],
                        sens.transpose(1, 2, 0)[..., None, :],
                    )
                except:
                    print("Calibration error, moving on to next file.")
                    continue

                # Save data along with sensitivity maps.
                # Target coil-combined images are [slices, X, Y], while sensitivity maps and k-space are [slices, coils, X, Y]
                h5f = h5py.File(output_dir + "/%d_%d.h5" % (case, sli), "w")
                h5f.create_dataset("filename", data=name)
                h5f.create_dataset("kspace", data=kspace_slice)
                h5f.create_dataset("sensmaps", data=sens)
                h5f.create_dataset("target", data=im_coil_combined)
                h5f.close()

        f.close()


def get_args():
    parser = OptionParser()
    # Will be split into train/validation
    parser.add_option(
        "--train",
        dest="load_train",
        default="/mikQNAP/NYU_knee_data/multicoil_train",
        help="Folder directory contains train dataset (h5 format)",
    )
    parser.add_option(
        "--test",
        dest="load_test",
        default="/mikQNAP/NYU_knee_data/multicoil_val",
        help="Folder directory contains test dataset (h5 format)",
    )
    parser.add_option(
        "--train_out",
        dest="target_train",
        default="/mikRAID/fredwang/full_knee_data/multicoil_train_processed",
        help="Target folder directory for processed train data",
    )
    parser.add_option(
        "--test_out",
        dest="target_test",
        default="/mikRAID/fredwang/full_knee_data/multicoil_test_processed",
        help="Target folder directory for processed test data",
    )
    parser.add_option(
        "-d",
        "--data",
        dest="data",
        default="knee",
        help="Which dataset to create sensitivity maps, fastMRI knee, fastMRI brain or other",
    )
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    fastMRI_brain_config = {
        "height": 640,
        "width": 320,
        "coils": 16,
        "label": ["AXT1", "AXT1PRE", "AXT1POST", "AXT2", "AXFLAIR"],
        "slices_keep": 10,
        "calib": 21,
    }
    fastMRI_knee_config = {
        "height": 640,
        "width": 372,
        "coils": 15,
        "label": ["CORPD_FBK"],
        "slices_keep": 10,
        "calib": 20,
    }
    # For other non-fastMRI datasets.
    other_config = {
        "height": 0,
        "width": 0,
        "coils": 0,
        "label": [],
        "slices_keep": 0,
        "calib": 0,
    }

    args = get_args()
    if args.data == "knee":
        config = fastMRI_knee_config
    elif args.data == "brain":
        config = fastMRI_brain_config
    elif args.data == "other":
        config = other_config
    else:
        raise ValueError(
            "Unknown data type specified. Must be 'knee', 'brain', or 'other'"
        )

    create_maps(args.load_train, args.target_train, config)
    create_maps(args.load_test, args.target_test, config)
