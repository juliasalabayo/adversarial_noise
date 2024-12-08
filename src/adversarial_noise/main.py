import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.transforms import ToPILImage

from adversarial_noise.adversarial_model import generate_adversarial_image
from adversarial_noise.utils import (
    get_image,
    get_model_categories,
    load_model,
)


class Config:
    """Class to manage and validate configuration settings."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = self._load_config()

        # Default types for validation
        self.expected_types = {
            "image_path": {"type": str},
            "output_path": {"type": str},
            "epsilon": {"type": float, "range": (0, 1)},
            "model_name": {"type": str},
            "image_range": {"type": list, "length": 2},
        }

    def _load_config(self) -> dict:
        """Load the YAML configuration file."""
        with Path.open(self.config_path) as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Any | None = None) -> Any:
        """Get a value from the config."""
        return self.config.get(key, default)


def main() -> None:
    """Add adversarial noise to an image and save output."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate an adversarial image."
    )
    parser.add_argument(
        "input_file_path",
        type=str,
        help="Name of the input image file (including path)",
    )
    parser.add_argument(
        "category",
        type=str,
        help="Target category",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the configuration file (default: config.yml)",
    )
    args = parser.parse_args()

    # Load and validate configuration
    config = Config(args.config)

    # Extract configuration values
    image_path = Path(args.input_file_path)
    target_category = args.category
    epsilon = config.get("epsilon")
    model_name = config.get("model_name")
    image_range = config.get("image_range")
    output_path = Path(config.get("output_path"))

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and categories
    model = load_model(model_name, device)
    categories = get_model_categories(model_name)

    # Load and preprocess image
    image_tensor = get_image(image_path)

    # Generate adversarial image
    adversarial_tensor = generate_adversarial_image(
        model,
        image_tensor,
        target_category,
        categories,
        device,
        epsilon,
        image_range,
    )

    # Save adversarial image
    output_path.mkdir(parents=True, exist_ok=True)
    input_file_name = (
        args.input_file_path.replace("/", "_")
        .split(".png")[0]
        .split("data_")[1]
    )
    output_file = (
        output_path
        / f"adversarial_to_{target_category}_{input_file_name}_{epsilon}.png"
    )
    adversarial_image = ToPILImage()(adversarial_tensor.squeeze(0).cpu())
    adversarial_image.save(output_file)
    print(f"Adversarial image saved to {output_file}")  # noqa: T201


if __name__ == "__main__":
    main()
