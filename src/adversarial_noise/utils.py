from typing import Literal

import torch
from PIL import Image
from torchvision import models, transforms

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


def convert_category_to_tensor(
    target_category: str, categories: list[str], device: torch.device
) -> torch.Tensor:
    """
    Convert a label category to its tensor representation.

    Parameters
    ----------
        target_category: Target classification label.
        categories: List of available labels.

    Returns
    -------
        torch.Tensor: Category tensor
    """
    if target_category not in categories:
        msg = f"Category {target_category} not found in categories."
        raise ValueError(msg)
    target_tensor = torch.tensor(
        [categories.index(target_category)], dtype=torch.long
    )

    return target_tensor.to(device)


def prepare_image(image: Image) -> torch.Tensor:
    """
    Tranform an image to be interpreted by ResNet models.

    It uses centercrop to preserve the aspect ratio, preferred in some models,
    like ResNet.

    Parameters
    ----------
        image: Image to prepare for inputting to the models.

    Returns
    -------
        torch.Tensor: Processed image.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    # Convert image to tensor
    input_tensor = transform(image)
    # Add batch dimension, needed by the model
    input_batch = input_tensor.unsqueeze(0)

    # Ensure tensor is of type float32
    return input_batch.to(torch.float32)


def get_image(
    image_path: str,
    colour_scale: Literal["RGB", "Grayscale", "CYMK", "RGBA"] = "RGB",
) -> torch.Tensor:
    """
    Load image from a file and process it for ResNet use.

    Parameters
    ----------
        image_path: Path to the image file.

    Returns
    -------
        Image: Loaded image.
    """
    image = Image.open(image_path).convert(colour_scale)
    return prepare_image(image)
