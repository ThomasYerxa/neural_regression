from typing import Callable, Optional

import numpy as np
from torchvision import transforms as T

class ApplyCosineAperture:
    def __init__(
        self,
        img_size: int = 320,
        input_degrees: int = 4,
        output_degrees: int = 8,
        aperture_degrees: int = 4,
        gray_c: int = 128,
        input_channels: int = 1,
    ):
        self.gray_c = gray_c
        self.input_degrees = input_degrees
        self.aperture_degrees = aperture_degrees
        self.pos = np.array([0, 0])
        self.output_degrees = output_degrees
        self.size_px = np.array([img_size, img_size])

        # Image size
        px_deg = self.size_px[0] / self.input_degrees

        self.size_px_out = (
            self.size_px * (self.output_degrees / self.input_degrees)
        ).astype(int)
        cnt_px = (self.pos * px_deg).astype(int)

        size_px_disp = ((self.size_px_out - self.size_px) / 2).astype(int)

        self.fill_ind = [
            [
                (size_px_disp[0] + cnt_px[0]),
                (size_px_disp[0] + cnt_px[0] + self.size_px[0]),
            ],
            [
                (size_px_disp[1] + cnt_px[1]),
                (size_px_disp[1] + cnt_px[1] + self.size_px[1]),
            ],
        ]

        # Image aperture
        a = self.aperture_degrees * px_deg / 2
        # Meshgrid with pixel coordinates
        x = np.arange(self.size_px_out[1]) - self.size_px_out[1] / 2
        y = np.arange(self.size_px_out[0]) - self.size_px_out[0] / 2
        xv, yv = np.meshgrid(x, y)
        # Raised cosine aperture
        inner_mask = (xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2 < a**2
        cos_mask = (
            1
            / 2
            * (
                1
                + np.cos(
                    np.sqrt((xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2) / a * np.pi
                )
            )
        )
        cos_mask[np.logical_not(inner_mask)] = 0

        self.cos_mask = cos_mask

    def __call__(self, im):
        im = im - self.gray_c * np.ones(self.size_px)
        im_template = np.zeros(self.size_px_out)

        im_template[
            self.fill_ind[0][0] : self.fill_ind[0][1],
            self.fill_ind[1][0] : self.fill_ind[1][1],
        ] = im
        im_masked = (im_template * self.cos_mask) + self.gray_c * np.ones(
            self.size_px_out
        )

        return im_masked.astype(np.float32)


class FreemanZiembaV1V2_ImageNet_Transform:
    """
    Transformation class for FreemanZiembaV1V2 dataset images to be compatible with ImageNet-pretrained models.
    The transformations are as follows:
        (1) Apply a raised cosine aperture to the images to match the original experimental conditions.
        (2) Convert grayscale images to 3-channel RGB by duplicating the single channel.
        (3) Normalize images using ImageNet mean and standard deviation. 

    Parameters
    ----------
    img_size : int, optional, default=320
        The size of the input images (img_size x img_size).
    """
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.transform = T.Compose(
            [
                ApplyCosineAperture(
                    img_size=320,
                    input_degrees=4,
                    output_degrees=8,
                    aperture_degrees=4,
                ),
                T.ToPILImage(),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Resize((img_size, img_size)),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img):
        return self.transform(img)

class LieberV4_ImageNet_Transform:
    """
    Transformation class for LieberV4 dataset images to be compatible with ImageNet-pretrained models.
    The transformations are as follows:
        (1) Rescale images to match resolution expected by the model:
                - By default ImageNet models expect 224x224 images, and a good rule of thumb is to assume
                  these images span ~8 degrees of visual angle.
                - The images presented in the in the Lieber V4 dataset spanned 6.4 degrees of visual angle,
                -> So, we resize images to  (6.4/8)*224 = 179 pixels.
        (2) Convert grayscale images to 3-channel RGB by duplicating the single channel.
        (3) Normalize images using ImageNet mean and standard deviation. 

    Parameters
    ----------
    img_size : int, optional, default=179
        The size to which images will be resized (img_size x img_size).
    """
    def __init__(self, img_size: int = 179):
       self.img_size = img_size
       self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.img_size, self.img_size)),
                T.Grayscale(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img):
        return self.transform(img)