"""Recon object for combining system blocks (such as datasets and transformers),
model blocks (such as CNNs and ResNets), and optimization blocks (such as conjugate
gradient descent)."""

# !/usr/bin/env python

import numpy as np
import pytorch_lightning as pl
import scipy.signal
import torch
from deepinpy.forwards import MultiChannelMRIDataset
from deepinpy.utils import utils
from torchvision.utils import make_grid

from kband.utils_pipeline import band_mask


@torch.jit.script
def calc_nrmse(gt, pred):
    """Calculates NRMSE between input tensors gt and pred.

    Args:
        gt (Tensor): Ground truth image tensor.
        pred (Tensor): Reconstructed image tensor.

    Returns:
        (Tensor) NRMSE between gt and pred.
    """
    temp = torch.sqrt(torch.mean(torch.square(torch.abs(gt) - torch.abs(pred))))
    rr = torch.max(torch.abs(gt)) - torch.min(torch.abs(gt))  # range
    return torch.div(temp, rr)


def fft2c(_tensor):
    """Returns the center shifted FFT of the input tensor

    Args:
        _tensor (Tensor): tensor to center shift FFT

    Returns:
        (Tensor) center shifted FFT of _tensor.
    """
    _tensor = torch.fft.ifftshift(_tensor, dim=(-2, -1))
    _tensor = torch.fft.fft2(_tensor)
    return torch.fft.fftshift(_tensor, dim=(-2, -1))


def ifft2c(_tensor):
    """Returns the center shifted IFFT of the input tensor

    Args:
        _tensor (Tensor): tensor to center shift IFFT

    Returns:
        (Tensor) center shifted IFFT of _tensor.
    """
    _tensor = torch.fft.fftshift(_tensor, dim=(-2, -1))
    _tensor = torch.fft.ifft2(_tensor)
    return torch.fft.ifftshift(_tensor, dim=(-2, -1))


def generate_W_mask(height=400, width=300, angles=tuple(range(180)), R_band=6.0):
    """Generates a density compensation mask, assuming a uniform distribution over k-band angles provided.
    Method: 
        We initialize a zeros mask, then iterative over all possible bands and add ones in all the pixels that are included in the bands.
        And in the end we invert this matrix to get the density compensation matrix.
    Args:
        height (int): number of rows in the density compensation mask (default 640)
        width (int): number of columns in the density compensation mask (default 372)
        angles (list): which k-band angles are used (default all angles between 0 and 180 degrees)
        R_band(double): R_band is the relative ratio that the band takes out of the full k-space size. For example, 
            if the k-space is 100x100 pixels, R_band=2, and the band is aligned with the y-axis (angle 0),
            then the band size will be 100x50. However, the angle can be arbitrarily chosen from 0 to 180.
            In our implementation, R_band must be higher than 1 and smaller than 10, i.e. the band can cover 
            anything between 100% of kspace to 10% of kspace. We avoid very narrow bands since a 
            fully sampled calibration area is required in kspace center for computing the sensitivity maps.
    Returns:
        loss_mask (ndarray): Density compensation mask to be applied during loss.
    """
    assert R_band >= 1, "R_band must be greater than or equal to 1."
    assert R_band <= 10, "R_band must be less than or equal to 10."

    W_mask = np.zeros((height, width))
    for angle in angles:
        # Create mask with dimensions larger than necessary to account for rotation.
        W_mask += band_mask(height, width, angle, R_band)

    # Replace 0 values with infinity to avoid divide by zero errors
    W_mask = np.where(W_mask < 0.5, np.inf, W_mask)

    W_mask = 1 / W_mask
    W_mask *= np.count_nonzero(W_mask) / np.sum(W_mask)

    # Smoothing filter to remove extremely high values from poor interpolation.
    f = np.ones((5, 5)) / 25
    W_mask = scipy.signal.convolve2d(W_mask, f, mode="same")

    return W_mask


class Recon(pl.LightningModule):
    """An abstract class for implementing system-model-optimization (SMO) constructions.
    Modified to support k-band, with the addition of the W mask and custom loss functions.

    The Recon is an abstract class which outlines common functionality for all SMO structure implementations. All of them share hyperparameter initialization, MCMRI dataset processing and loading, loss function, training step, and optimizer code. Each implementation of Recon must provide batch, forward, and get_metadata methods in order to define how batches are created from the data, how the model performs its forward pass, and what metadata the user should be able to return. Currently, Recon automatically builds the dataset as an MultiChannelMRIDataset object; overload _build_data to circumvent this.

    Args:
        hprams (dict): Key-value pairings with parameter names as keys.

    Attributes:
        hprams (dict): Key-value pairings with hyperparameter names as keys.
        _loss_fun (func): Set to use either torch.nn.MSELoss or _abs_loss_fun.
        D (MultiChannelMRIDataset): Holds the MCMRI dataset.

    """

    def __init__(self, hparams):
        super(Recon, self).__init__()

        self._init_hparams(hparams)
        self._build_data()
        self.scheduler = None
        self.log_dict = None
        self.val_log_dict = None
        self.W_mask = generate_W_mask(R_band=self.hparams.R_band)
        self.iter = 0

    def _init_hparams(self, hparams):
        self.save_hyperparameters(hparams)

        if hparams.abs_loss:
            self.loss_fun = self._abs_loss_fun
        else:
            self.loss_fun = self._loss_fun

    def _loss_fun(self, pred, gt, loss_mask, W, loss="kspace_L1"):
        """Abstraction for all three loss functions, to be called in the training and validation loop.

        Args:
            pred (Tensor): reconstructed tensor by the network of the image
            gt (Tensor): ground truth tensor of the image
            loss_mask (Tensor): specifies which points to calculate loss at. For k-band, this is the band mask.
            W (Tensor): density compensation mask
            loss (string): specifies loss function to use, see below for options. (default kspace_L1)

        Returns:
            Loss between pred and gt (masked by loss_mask and W)
        """

        if loss == "kspace_L1":
            return self._kband_l1_loss_fun(pred, gt, loss_mask, W)
        elif loss == "kspace_L1_no_W":
            return self._kband_l1_loss_no_W_fun(pred, gt, loss_mask)
        elif loss == "kspace_L2":
            return self._kband_l2_loss_fun(pred, gt, loss_mask, W)
        elif loss == "L2_image_full_supervision":
            return self._l2_loss_image_full_supervision(pred, gt)
        elif loss == "L1_kspace_full_supervision":
            return self._l1_loss_kspace_full_supervision(pred, gt)
        elif loss == "SSDU":
            return self._SSDU_loss_fun(pred, gt, loss_mask)
        elif loss == "SSDU_kband":
            raise NotImplementedError

    def _kband_l1_loss_fun(self, pred, gt, loss_mask, W):
        """
        Calculate L1 loss between input tensors pred and gt (ground truth) in k-space over the area specified by loss_mask and weighted by the W mask. 
        The loss mask specifies the pixels in which to calculate the loss at, and the W mask weights each pixel based on how frequently it is covered by a band.
        Used for k-band experiments.
        """
        resid = torch.abs(fft2c(pred) - fft2c(gt))
        resid = torch.mul(resid, W)
        resid = torch.mul(resid, loss_mask)
        return torch.mean(resid)

    def _kband_l1_loss_no_W_fun(self, pred, gt, loss_mask):
        """
        Calculate L1 loss between input tensors pred and gt (ground truth) in k-space over the area specified by loss_mask.
        The loss mask specifies the pixels in which to calculate the loss at.
        Used for k-band comparison without the W mask, and also for the square and vertical baselines.
        """
        resid = torch.abs(fft2c(pred) - fft2c(gt))
        resid = torch.mul(resid, loss_mask)
        return torch.mean(resid)

    def _kband_l2_loss_fun(self, pred, gt, loss_mask, W):
        """
        Calculate L2 loss between input tensors pred and gt (ground truth) in k-space over the area specified by loss_mask and weighted by W.
        The loss mask specifies the pixels in which to calculate the loss at.
        """
        resid = torch.abs(torch.square(fft2c(pred) - fft2c(gt)))
        resid = torch.mul(resid, W)
        resid = torch.mul(resid, loss_mask)
        return torch.mean(resid)

    def _l1_loss_kspace_full_supervision(self, pred, gt):
        """
        Calculate k-space L1-loss between the input tensors pred and gt (ground truth). 
        Notice that calculation is performed for all the pixels in k-space, because we do not use any mask here. 
        This corresponds to full supervision using all the k-space values. Notice also that we do not use the W_mask here.
        Used for our comparisons with the MoDL fully supervised baseline.
        """
        resid = torch.abs(fft2c(pred) - fft2c(gt))
        return torch.mean(resid)

    def _l2_loss_image_full_supervision(self, pred, gt):
        """
        Calculate image domain L2 loss between input tensors pred and gt (ground truth).
        Notice that calculation is performed for all the pixels in k-space, because we do not use any mask here. 
        This corresponds to full supervision using all the k-space values. Notice also that we do not use the W_mask here.
        """
        resid = torch.abs(torch.square(pred - gt))
        return torch.mean(resid)

    def _SSDU_loss_fun(self, pred, gt, loss_mask):
        """
        Calculate SSDU (mixed L1/L2) loss between input tensors pred and gt (ground truth) over the area specified by loss_mask. 
        This is based on the SSDU paper (Yaman et al., 2020, MRM). Used for our SSDU comparison experiments.
        """
        gt = torch.mul(fft2c(gt), loss_mask)
        pred = torch.mul(fft2c(pred), loss_mask)
        resid = 0.5 * torch.norm(pred - gt) / torch.norm(gt) + 0.5 * torch.norm(
            pred - gt, p=1
        ) / torch.norm(gt, p=1)
        return torch.mean(resid)

    def _build_data(self):
        """Creates training and validation datasets. Automatically called by pytorch lightning."""
        # Training data.
        self.D = MultiChannelMRIDataset(
            data_file=self.hparams.data_train_file,
            masks_file=self.hparams.masks_train_file,
            stdev=self.hparams.stdev,
            num_data_sets=self.hparams.num_train_data_sets,
            adjoint_data=self.hparams.adjoint_data,
            id=0,
            clear_cache=False,
            cache_data=False,
            scale_data=False,
            fully_sampled=self.hparams.fully_sampled,
            data_idx=None,
            inverse_crime=self.hparams.inverse_crime,
            noncart=self.hparams.noncart,
        )
        # Validation data.
        self.V = MultiChannelMRIDataset(
            data_file=self.hparams.data_val_file,
            masks_file=self.hparams.masks_val_file,
            stdev=self.hparams.stdev,
            num_data_sets=self.hparams.num_val_data_sets,
            adjoint_data=self.hparams.adjoint_data,
            id=0,
            clear_cache=False,
            cache_data=False,
            scale_data=False,
            fully_sampled=self.hparams.fully_sampled,
            data_idx=None,
            inverse_crime=self.hparams.inverse_crime,
            noncart=self.hparams.noncart,
        )

    def _abs_loss_fun(self, x_hat, imgs):
        raise NotImplementedError

    def batch(self, data):
        """Not implemented, should define a forward operator A and the adjoint matrix of the input x.

        Args:
            data (Tensor): The data which the batch will be drawn from.

        Raises:
                NotImplementedError: Method needs to be implemented.
        """

        raise NotImplementedError

    def forward(self, y):
        """Not implemented, should perform a prediction using the implemented model.

        Args:
                y (Tensor): The data which will be passed to the model for processing.

        Returns:
            The model’s prediction in Tensor form.

        Raises:
                NotImplementedError: Method needs to be implemented.
        """

    def get_metadata(self):
        """Accesses metadata for the Recon.

        Returns:
            A dict holding the Recon’s metadata.

        Raises:
                NotImplementedError: Method needs to be implemented.
        """
        raise NotImplementedError

    def log_metadata(self, log_dict, key, fun=None):
        try:
            val = self.get_metadata()[key]
            if fun is not None:
                val = fun(val)
            log_dict[key] = val
        except KeyError:
            pass
        return log_dict

    def validation_step(self, batch, batch_idx):
        """Used automatically in validation loop by pytorch lightning.
        Returns a list of validation loss, NRMSE.

        Args:
            batch (tuple): Holds the indices of data and the corresponding data, in that order.
            batch_idx (None): Currently unimplemented.

        Returns:
            A list of validation loss and NRMSE.
        """
        idx, data = batch
        imgs = data["imgs"]
        ksp_cc = data["out"]

        self.batch(data)
        x_hat = self.forward(ksp_cc)

        if self.hparams.self_supervised:
            pred = self.A.forward(x_hat)
            gt = ksp_cc
        else:
            pred = x_hat
            gt = imgs

        loss_mask = data["loss_masks"]

        loss = self.loss_fun(
            pred,
            gt,
            torch.tensor(loss_mask).float().cuda(),
            torch.tensor(self.W_mask).float().cuda(),
            self.hparams.loss_function,
        )
        _loss = loss.clone().detach().requires_grad_(False)
        _loss_mask = torch.tensor(loss_mask).float().cuda()

        # Here we compute the images obtained from the band only, for both ground truth and predicted (reconstructed) image.
        # This is done only for computing the validation error metric NRMSE. It is not used during the network training.
        band_image_pred = ifft2c(torch.mul(fft2c(pred), _loss_mask))
        band_image_gt = ifft2c(torch.mul(fft2c(gt), _loss_mask))
        NRMSE = calc_nrmse(band_image_pred, band_image_gt)
        _NRMSE = NRMSE.clone().detach().requires_grad_(False)

        return [_loss, _NRMSE]

    def validation_epoch_end(self, batch_parts):
        """Logs validation loss and NRMSE to tensorboard. Called automatically by pytorch lightning.

        Args:
            batch_parts (Tensor): concatenation of validation_step outputs over all validation data.
        """
        self.logger.experiment.add_scalar(
            "validation_loss",
            torch.mean(torch.stack([i[0] for i in batch_parts])),
            self.global_step,
        )
        self.logger.experiment.add_scalar(
            "validation_NRMSE",
            torch.mean(torch.stack([i[1] for i in batch_parts])),
            self.global_step,
        )

    # FIXME: batch_nb parameter appears unused.
    def training_step(self, batch, batch_nb):
        """Defines a training step solving deep inverse problems, including batching, performing a forward pass through
        the model, and logging data. This may either be supervised or unsupervised based on hyperparameters.

        Args:
            batch (tuple): Should hold the indices of data and the corresponding data, in said order.
            batch_nb (None): Currently unimplemented.

        Returns:
            A dict holding performance data and current epoch for performance tracking over time.
        """
        idx, data = batch
        idx = utils.itemize(idx)
        imgs = data["imgs"]
        ksp_cc = data["out"]

        self.batch(data)

        x_hat = self.forward(ksp_cc)

        if self.logger and (
            self.current_epoch % self.hparams.save_every_N_epochs == 0
            or self.current_epoch == self.hparams.num_epochs - 1
        ):
            _b = ksp_cc.shape[0]
            if _b == 1 and idx == 0:
                _idx = 0
            elif _b > 1 and 0 in idx:
                _idx = idx.index(0)
            else:
                _idx = None
            if _idx is not None:
                with torch.no_grad():
                    if self.x_adj is None:
                        x_adj = self.A.adjoint(ksp_cc)
                    else:
                        x_adj = self.x_adj

                    _x_hat = utils.t2n2(x_hat[_idx, ...])
                    _x_gt = utils.t2n2(imgs[_idx, ...])
                    _x_adj = utils.t2n2(x_adj[_idx, ...])

                    if len(_x_hat.shape) > 2:
                        _d = tuple(range(len(_x_hat.shape) - 2))
                        _x_hat_rss = np.linalg.norm(_x_hat, axis=_d)
                        _x_gt_rss = np.linalg.norm(_x_gt, axis=_d)
                        _x_adj_rss = np.linalg.norm(_x_adj, axis=_d)

                        myim = torch.tensor(
                            np.stack((_x_adj_rss, _x_hat_rss, _x_gt_rss), axis=0)
                        )[:, None, ...]
                        grid = make_grid(
                            myim, scale_each=True, normalize=True, nrow=8, pad_value=10
                        )
                        self.logger.experiment.add_image(
                            "3_train_prediction_rss", grid, self.current_epoch
                        )

                        while len(_x_hat.shape) > 2:
                            _x_hat = _x_hat[0, ...]
                            _x_gt = _x_gt[0, ...]
                            _x_adj = _x_adj[0, ...]

                    myim = torch.tensor(
                        np.stack((np.abs(_x_hat), np.angle(_x_hat)), axis=0)
                    )[:, None, ...]
                    grid = make_grid(
                        myim, scale_each=True, normalize=True, nrow=8, pad_value=10
                    )
                    self.logger.experiment.add_image(
                        "2_train_prediction", grid, self.current_epoch
                    )

                    if self.current_epoch == 0:
                        myim = torch.tensor(
                            np.stack((np.abs(_x_gt), np.angle(_x_gt)), axis=0)
                        )[:, None, ...]
                        grid = make_grid(
                            myim, scale_each=True, normalize=True, nrow=8, pad_value=10
                        )
                        self.logger.experiment.add_image("1_ground_truth", grid, 0)

                        myim = torch.tensor(
                            np.stack((np.abs(_x_adj), np.angle(_x_adj)), axis=0)
                        )[:, None, ...]
                        grid = make_grid(
                            myim, scale_each=True, normalize=True, nrow=8, pad_value=10
                        )
                        self.logger.experiment.add_image("0_input", grid, 0)

        if self.hparams.self_supervised:
            pred = self.A.forward(x_hat)
            gt = ksp_cc
        else:
            pred = x_hat
            gt = imgs

        loss_mask = data["loss_masks"]

        loss = self.loss_fun(
            pred,
            gt,
            torch.tensor(loss_mask).float().cuda(),
            torch.tensor(self.W_mask).float().cuda(),
            loss=self.hparams.loss_function,
        )
        _loss = loss.clone().detach().requires_grad_(False)

        try:
            _lambda = self.l2lam.clone().detach().requires_grad_(False)
        except:
            _lambda = 0
        _epoch = self.current_epoch
        _nrmse = calc_nrmse(imgs, x_hat).detach().requires_grad_(False)

        log_dict = {
            "lambda": _lambda,
            "train_loss": _loss,
            "epoch": self.current_epoch,
            "nrmse": _nrmse,
            "val_loss": 0,
        }

        # FIXME: let the user specify this list
        log_dict = self.log_metadata(log_dict, "num_cg", fun=np.max)
        keys_list = ["mean_residual_norm", "mean_eps"]
        for key in keys_list:
            log_dict = self.log_metadata(log_dict, key)

        if self.logger:
            for key in log_dict.keys():
                self.logger.experiment.add_scalar(key, log_dict[key], self.global_step)

        self.log_dict = log_dict
        return loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        if self.log_dict:
            for key in self.log_dict.keys():
                if type(self.log_dict[key]) == torch.Tensor:
                    items[key] = utils.itemize(self.log_dict[key])
                else:
                    items[key] = self.log_dict[key]
        return items

    def configure_optimizers(self):
        """Determines whether to use Adam or SGD depending on hyperparameters.

        Returns:
            Torch’s implementation of SGD or Adam, depending on hyperparameters.
        """

        if "adam" in self.hparams.solver:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.step)
        elif "sgd" in self.hparams.solver:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.step)
        if self.hparams.lr_scheduler != -1:
            # doing self.scheduler will create a scheduler instance in our self object
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hparams.lr_scheduler[0],
                gamma=self.hparams.lr_scheduler[1],
            )
        if self.scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        """Creates a DataLoader object, with distributed training if specified in the hyperparameters.

        Returns:
            A PyTorch DataLoader that has been configured according to the hyperparameters.
        """

        return torch.utils.data.DataLoader(
            self.D,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=0,
            drop_last=True,
        )

    def val_dataloader(self):
        """Creates a DataLoader object, with distributed training if specified in the hyperparameters.

        Returns:
            A PyTorch DataLoader that has been configured according to the hyperparameters.
        """

        return torch.utils.data.DataLoader(
            self.V,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=0,
            drop_last=True,
        )
