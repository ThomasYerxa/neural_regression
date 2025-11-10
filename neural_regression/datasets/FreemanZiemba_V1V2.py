from typing import Callable, Optional, Tuple, Union
from pathlib import Path
import os
import scipy.io
import numpy as np
import torch
from itertools import product
from PIL import Image 


class FreemanZiembaV1V2(torch.utils.data.Dataset):
    """
    Freeman+Ziemba V1 and V2 Datasets: Monkey V1/V2 responses to naturalistic textures and spectral noise counterparts.

    Parameters
    ----------
    brain_area : str
        'V1' or 'V2' to specify the brain area dataset to load.
    data_directory : str or Path, optional
        Directory where the dataset is stored. Default is '/mnt/ceph/users/xzhao/Datasets/FreemanZiemba2013'.
        Images are in subdirectory 'stim' and neural data in 'data'.
    image_transform : callable, optional
        A function/transform that takes in an image and returns a transformed version. Default is None.
    neural_transform : callable, optional
        A function/transform that takes in neural data and returns a transformed version. Default is None.
    to_numpy : bool, optional
        If True, images are returned as numpy arrays
    """
    def __init__(
        self,
        brain_area: str,
        data_directory: Union[str, Path] = "/mnt/ceph/users/xzhao/Datasets/FreemanZiemba2013",
        image_transform: Optional[Callable] = None,
        neural_transform: Optional[Callable] = None,
        to_numpy: Optional[bool] = False,
    ):

        super().__init__()
        self.brain_area = brain_area
        self.data_directory = data_directory
        self.stim_directory = os.path.join(self.data_directory, "stim")
        self.response_directory = os.path.join(self.data_directory, "data")
        self.image_transform = image_transform
        self.neural_transform = neural_transform
        self.to_numpy = to_numpy

        # Create list of stimulus parameters
        self.image_families = [
            327, 336, 393, 402, 13, 18, 23, 30,
            38, 48, 52, 56,  60, 71, 99,
        ]
        self.stimulus_types = ["noise", "tex"]
        self.sample_indices = list(range(1, 16))

        # full list of stim params: 15x15x2 = 450
        self.stim_params = list(product(self.stimulus_types, self.image_families, self.sample_indices))

        # Load neural data
        self.neural_responses, self.mean_responses, self.self_consistency = self._load_neural_data(brain_area)


    def _load_neural_data(self, brain_area):
        # load data
        neural_data = scipy.io.loadmat(
            os.path.join(self.response_directory, "FreemanZiemba2013_V1V2data.mat")
        )
        neural_data = neural_data["V1V2_data"]

        V1_data = neural_data[0, :-1]
        V2_data = neural_data[1, :]

        V1_mean_responses = np.mean(V1_data, axis=-1)
        V2_mean_responses = np.mean(V2_data, axis=-1)

        # load self similarity (ceiling)
        V1_self_consistency = np.load(
            "/mnt/home/tyerxa/repos/local_low_dimensionality/V1_self_consistency.npy"
        )
        V2_self_consistency = np.load(
            "/mnt/home/tyerxa/repos/local_low_dimensionality/V2_self_consistency.npy"
        )

        if brain_area == "V1":
            data = V1_data
            mean_responses = V1_mean_responses
            self_consistency = V1_self_consistency
        elif brain_area == "V2":
            data = V2_data
            mean_responses = V2_mean_responses
            self_consistency = V2_self_consistency

        return data, mean_responses, self_consistency


    def _parse_stim_file(self, stim_file):
        name_parts = stim_file.split("-")
        stim_type = name_parts[0]
        im_index = int(name_parts[2][2:])
        sample_index = int(name_parts[3][3:-4])

        return stim_type, im_index, sample_index

    def _gen_stim_file(self, stim_type, im_index, sample_index):
        return f"{stim_type}-320x320-im{im_index}-smp{sample_index}.png"

    def __len__(self):
        return len(self.stim_params)

    def _get_image(self, idx):
        stim_type, im_index, sample_index = self.stim_params[idx]
        stim_file = self._gen_stim_file(
            stim_type=stim_type, im_index=im_index, sample_index=sample_index
        )
        stim_path = os.path.join(self.stim_directory, stim_file)
        stim = Image.open(stim_path)
        if self.to_numpy:
            stim = np.array(stim)
        if self.image_transform:
            stim = self.image_transform(stim)

        return stim
    
    def _get_mean_response(self, idx):
        stim_type, im_index, sample_index = self.stim_params[idx]
        stim_type_int = 0 if stim_type == "noise" else 1
        neural_response = self.neural_responses[
            :,
            self.image_families.index(im_index),
            stim_type_int,
            sample_index - 1,
        ]
        mean_response = np.mean(neural_response, axis=-1)

        if self.neural_transform:
            mean_response = self.neural_transform(mean_response)
        return mean_response
    
    def _get_stim_params(self, idx):
        stim_type, im_index, sample_index = self.stim_params[idx]
        stim_type_int = 0 if stim_type == "noise" else 1
        return {"stim_type": stim_type_int, "im_index": im_index, "sample_index": sample_index}

    def __getitem__(self, idx):
        img = self._get_image(idx)
        resp = self._get_mean_response(idx)
        params = self._get_stim_params(idx)
        return img, resp, params

