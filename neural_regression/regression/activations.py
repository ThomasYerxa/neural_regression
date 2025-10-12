from typing import List, Tuple, Optional

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import torch
import numpy as np

def get_activations(
    model: torch.nn.Module, 
    result_layers: List[str], 
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = torch.device("cuda")
):
    """
    Get activations from specified layers of a model for all images in a dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model from which to extract activations.
    result_layers : List[str]
        List of layer names from which to extract activations.
    loader : torch.utils.data.DataLoader
        DataLoader providing batches of images and associated data. 
    device : torch.device, optional
        Device to run the model on (default is CUDA).

    Returns
    -------
    response_dict : dict
        Dictionary mapping layer names to their corresponding activations.
    neural_responses : np.ndarray
        Array of neural responses corresponding to the images.
    params_dict : dict
        Dictionary of stimulus parameters associated with the images.
    """
    return_layers = {k: k for k in result_layers}
    model_mg = MidGetter(model.eval().to(device), return_layers=return_layers, keep_output=False)
    response_dict = {}
    params_dict = {}
    neural_responses = []
    for i, (images, neural_response, stim_params) in enumerate(loader):
        with torch.no_grad():
            images = images.to(device)
            outs, _ = model_mg(images)
        
        for k, v in outs.items():
            if k not in response_dict:
                response_dict[k] = []
            response_dict[k].append(v.cpu().numpy())
        
        for k, v in stim_params.items():
            if k not in params_dict:
                params_dict[k] = []
            params_dict[k].append(v.numpy())

        neural_responses.append(neural_response.numpy())

    for k, v in response_dict.items():
        response_dict[k] = np.concatenate(v, axis=0)
    
    for k, v in params_dict.items():
        params_dict[k] = np.concatenate(v, axis=0)

    neural_responses = np.concatenate(neural_responses, axis=0)
    
    return response_dict, neural_responses, params_dict

    
