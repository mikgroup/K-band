# [K-band: Self-supervised MRI Reconstruction via Stochastic Gradient Descent over K-space Subsets](https://arxiv.org/abs/2308.02958)

## Installation Instructions:

`git clone https://github.com/mikgroup/Kband-recon.git`

Installing dependencies:
```
pip install bart
pip install sigpy
pip install tensorboard
conda install -c pytorch pytorch 
pip install -r deepinpy/requirements.txt
pip install .  # within Kband-recon directory
```

## Acknowledgements:
DeepInPy code obtained from https://github.com/utcsilab/deepinpy (Tamir et al., 2020, ISMRM). 

Signal processing utility functions provided by SigPy (https://sigpy.readthedocs.io/en/latest/) (Ong et al., 2019, ISMRM) and BART (http://mrirecon.github.io/bart/) (Uecker et al. 2016, ISMRM). 

Dataset provided by FastMRI (https://fastmri.org/) (Knoll et al., 2020, Radiology).

## Description:

K-band provides a novel method for training MRI reconstruction networks on limited-resolution data. This allows for training data to be efficiently acquired in k-space bands, as this can speed up acquisition while reducing artifacts. Importantly, the band orientation changes randomly across examples, hence during training the network is exposed to all k-space areas. During inference, the network can reconstruct full-resolution images.

## Data Pipeline

The scripts in `kband/` simulate low-resolution band acquisition from the fully sampled fastMRI database in preparation for training k-band and other comparison methods such as MoDL and SSDU. Example usage of these scripts can be seen below:

### Sensitivity Map Generation + Data Filtering
```
python kband/generate_maps.py --train [train data path] --test [test data path] --train_out [processed train data output path] --test_out [processed test data output path] -d [knee/brain/other]
```
For knee data: Filters proton density knee data of dimensions 640x372 with top/bottom 8 slices removed, and the calculates sensitivity maps using bart.

For brain data: Filters out brain data of dimensions 640x320 with 16 coil measurements per scan, and the calculates sensitivity maps using bart. 

For other data: specify config parameters (height, width, etc.) inside generate_maps.py.

### Input Data Generation

```
python kband/generate_data.py --config [knee/brain/other]
```
Generates training, validation, and test data files. Height/width for cropping and path to load/save data can be adjusted in the config dict inside generate_data.py. The generated h5 data will consist of the following fields:

```
ksp: [Nsamples, Ncoils, NX, NY]
imgs: [Nsamples, NX, NY]
maps: [Nsamples, Ncoils, NX, NY]
```

### Input Mask Generation

```
python kband/generate_masks.py --config [knee/brain/other] --type [kband/modl/ssdu/square/vertical] --undersampling [2d/1d]
```
Generates training, validation, and test mask files. Height/width/undersampling rate and path to save data can be adjusted in the config dict inside generate_data.py. Undersampling method can be chosen from 2D or 1D Poisson disc sampling. The generated h5 masks will consist of the following fields:

```
masks: [Nsamples, NX, NY]
loss_masks: [Nsamples, NX, NY]
```

# Running K-band

DeepInPy (slightly modified from the original) provides our training framework. In order to run expirements, run 

```
python deepinpy/main.py --config {some/path/to/config.json}
```

The config file consists of several important arguments:

```
"data_train_file": path to training data (h5 format)
"data_val_file": path to validation data (h5 format)
"data_inference_file": path to inference data (h5 format)
"masks_train_file": path to training masks (h5 format)
"masks_val_file": path to validation masks (h5 format)
"masks_inference_file": path to inference masks (h5 format)
"name": experiment name to save checkpoints and tensorboard logging
"recon": "modl"
"network": "ResNet"
"num_blocks": number of residual blocks
"num_unrolls": number of unrolls per block
"num_train_data_sets": 1600
"num_val_data_sets": 400
"num_inference_data_sets": 400
"loss_function": Can be chosen from kspace_L1 (k-band),L1_kspace_full_supervision (MoDL), SSDU (SSDU).
"step": 0.0001
"num_epochs": 10,
"lr_scheduler": [5, 0.1],
"gpu": Enables GPU.
```

Some example configs for our experiments can be found in `deepinpy/configs`.

# Citation

If you use our code, please cite:

```
@inproceedings{wang2023kband,
   title={{K-band: Training self-supervised reconstruction networks using limited-resolution data}},
  author={F. Wang and Qi, Han and De Goyeneche, Alfredo, and Lustig, Michael and E. Shimron},
  booktitle={Proceedings of the ISMRM Workshop on Data Sampling and Imaging Reconstruction, Sedona},
  year={2023.}
},
@inproceedings{qi2023kband,
   title={{K-band: Training self-supervised reconstruction networks using limited-resolution data}},
  author={H. Qi and Wang, F. and  De Goyeneche, Alfredo, and Lustig, Michael and E. Shimron},
  booktitle={Proceedings of the ISMRM Annual Meeting, Toronto},
  year={2023.}
},
```
