import pytest
import torch
from adversarial_noise.adversarial_model import generate_adversarial_image
from adversarial_noise.utils import load_model


@pytest.mark.parametrize("epsilon", [0.01, 0.1])
def test_generate_adversarial_image(epsilon: float) -> None:
    """Test if the adversarial image generation function works."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("resnet50", device)
    categories = ["gibbon", "goldfish", "giant panda"]

    # Create a dummy image tensor
    image_tensor = torch.rand(1, 3, 224, 224).to(device)

    # Target category to misclassify the image
    target_category = categories[0]

    # Mock image clamp range
    image_range = [0, 1]

    adversarial_image = generate_adversarial_image(
        model,
        image_tensor,
        target_category,
        categories,
        device,
        epsilon,
        image_range,
    )

    # Adversarial image should be a 4D tensor
    assert adversarial_image.ndimension() == 4
    # Adversarial image should have the same shape as the input image
    assert adversarial_image.shape == image_tensor.shape
    # Adversarial image values should be in the range [0, 1]
    assert adversarial_image.min() >= min(image_range)
    assert adversarial_image.max() <= max(image_range)
