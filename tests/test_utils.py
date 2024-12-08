import pytest
import torch
from adversarial_noise.utils import load_model, prepare_image
from PIL import Image


@pytest.mark.parametrize("model_name", ["resnet50"])
def test_load_model(model_name: str) -> None:
    """Test if the load_model function loads a model correctly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, device)

    assert isinstance(
        model, torch.nn.Module
    ), "Model should be of type torch.nn.Module"
    assert (
        next(model.parameters()).device == device
    ), f"Model should be loaded to {device}"


def test_preprocess_image() -> None:
    """Test if the image preprocessing works correctly."""
    # Create a dummy image
    image = Image.new("RGB", (256, 256), color="red")
    image_tensor = prepare_image(image)

    # Output tensor should be 4D (batch size, channels, height, width)
    assert image_tensor.ndimension() == 4
    # Image should be resized to 224x224 with 3 channels
    assert image_tensor.shape[1:] == torch.Size([3, 224, 224])
    # Tensor should be of dtype float32
    assert image_tensor.dtype == torch.float32
