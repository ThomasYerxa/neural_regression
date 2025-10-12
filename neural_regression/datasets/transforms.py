from typing import Callable, Optional

from torchvision import transforms as T

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