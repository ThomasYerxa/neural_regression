from typing import Optional, Union
import os
from pathlib import Path

from torchvision import models
import torch

def get_alexnet_model(
    model_name: str,
    ext_model_path: Optional[Union[str, Path]] = "/mnt/ceph/users/tyerxa/ext_models/",
    device: Optional[torch.device] = torch.device("cpu"),
):
    if model_name == "supervised":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    if model_name == "random":
        model = models.alexnet(weights=None)
    if model_name == "robust":
        model = models.alexnet(weights=None)
        ckpt = torch.load(os.path.join(ext_model_path, "AlexNet-R.pt"))
        sd_processed = {}
        for k, v in ckpt["model"].items():
            if ("attacker" not in k) and ("model" in k):
                k_new = k[13:]
                if "last_layer" in k_new:
                    k_new = k_new.replace("last_layer", "classifier.6")
                sd_processed[k_new] = v
        model.load_state_dict(sd_processed)

    return model.eval().to(device)

