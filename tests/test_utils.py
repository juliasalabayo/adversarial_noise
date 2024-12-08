import pytest
import torch
from adversarial_noise.utils import load_model


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
