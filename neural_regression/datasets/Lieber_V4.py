from typing import Callable, Optional, Tuple, Union
from pathlib import Path
import scipy.io
import numpy as np
import torch

class LieberV4(torch.utils.data.Dataset):
    """
    LieberV4 Dataset: Monkey V4 responses to scrambled natural images.

    Parameters
    ----------
    data_directory : str or Path, optional
        Directory where the dataset is stored. Default is '/mnt/ceph/users/tyerxa/neural_data/V4_Scramble_Data/'.
    image_transform : callable, optional
        A function/transform that takes in an image and returns a transformed version. Default is None.
    neural_transform : callable, optional
        A function/transform that takes in neural data and returns a transformed version. Default is None.
    """
    def __init__(
        self,
        data_directory: Union[str, Path] = "/mnt/ceph/users/tyerxa/neural_data/V4_Scramble_Data/",
        image_transform: Optional[Callable] = None,
        neural_transform: Optional[Callable] = None,
    ):

        super().__init__()
        self.stim_directory = data_directory
        self.image_transform = image_transform
        self.neural_transform = neural_transform

        # prepare stimuli (images)
        images = scipy.io.loadmat(f"{data_directory}/scrambleExperimentImages.mat")[
            "scrambleExperimentImages"
        ]
        neural_data = scipy.io.loadmat(
            f"{data_directory}/scrambleExperimentNeural.mat"
        )["scrambleExperimentNeural"]

        images = images[0, 0][0]
        image_indices = neural_data[0, 0][4]
        self.images = images[:, :, :-1]  # exclude blank image


        # prepare neural data (and parameters) (also exclude blank image)
        self.mean_responses = neural_data[0, 0][0][:, :-1]
        self.zscored_responses = neural_data[0, 0][1][:, :-1]
        self.image_indices = neural_data[0, 0][4][:-1, :]
        self.image_families = image_indices[:, 0][:-1]
        self.pool_sizes = image_indices[:, 1][:-1]
        self.shift_indices = image_indices[:, 2][:-1]

    def __len__(self):
        return self.images.shape[2]

    def _get_image(self, idx):
        img = self.images[:, :, idx]
        if self.image_transform:
            img = self.image_transform(img)
        return img
    
    def _get_mean_response(self, idx):
        resp = self.mean_responses[:, idx]
        if self.neural_transform:
            resp = self.neural_transform(resp)
        return resp
    
    def _get_stim_params(self, idx):
        img_index = self.image_indices[idx]
        img_family, pool_size, shift_index = img_index
        param_dict = {
            "img_family": img_family,
            "pool_size": pool_size,
            "shift_index": shift_index,
        }
        return param_dict

    def __getitem__(self, idx):
        img = self._get_image(idx)
        resp = self._get_mean_response(idx)
        params = self._get_stim_params(idx)
        return img, resp, params
