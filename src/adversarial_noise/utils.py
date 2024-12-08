import torch
from torchvision import models

AVAILABLE_MODELS = {"resnet50": "ResNet50_Weights"}


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    Load a pre-trained model and prepare it for evaluation.

    Parameters
    ----------
        device: Target device (CPU or GPU) based on availability.
        model_name: Name of the model.

    Returns
    -------
        torch.nn.Module: Pre-trained model.
    """
    if model_name not in AVAILABLE_MODELS:
        msg = f"Model {model_name} is not supported yet."
        raise ValueError(msg)

    # Get model weights
    weights = getattr(models, AVAILABLE_MODELS[model_name]).DEFAULT

    # Get model for evaluation
    model = getattr(models, model_name)(weights=weights)
    model.eval()

    return model.to(device)


def get_model_categories(model_name: str) -> list[str]:
    """
    Get the available image categories for a pre-trained model.

    Parameters
    ----------
        model_name: Name of the model.

    Returns
    -------
        list[str]: Categories associated with the model.
    """
    if model_name not in AVAILABLE_MODELS:
        msg = f"Model {model_name} not available."
        raise ValueError(msg)

    return getattr(models, AVAILABLE_MODELS[model_name]).DEFAULT.meta[
        "categories"
    ]
