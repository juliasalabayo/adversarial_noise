import torch

from adversarial_noise.utils import convert_category_to_tensor


def calculate_loss_gradient(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the gradient of the loss of the input image in comparison to the
    target label.

    Args:
        model (torch.nn.Module): Pre-trained model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        target_tensor (torch.Tensor): Target category tensor.

    Returns
    -------
        torch.Tensor: Gradient of the loss.
    """
    # Enable gradient tracking
    image_tensor.requires_grad = True

    # Forward pass
    output = model(image_tensor)
    loss = torch.nn.functional.cross_entropy(output, target_tensor)

    # Backward pass
    model.zero_grad()  # clear stored gradients
    loss.backward()

    # Return the gradient of the image tensor
    return image_tensor.grad.data


def apply_perturbation(
    image_tensor: torch.Tensor,
    gradient: torch.Tensor,
    epsilon: float,
    image_range: list[float, float],
) -> torch.Tensor:
    """
    Apply adversarial perturbation to the image.

    Parameters
    ----------
        image_tensor: Original image tensor.
        gradient: Gradient of the loss.
        epsilon: Perturbation strength.

    Returns
    -------
        torch.Tensor: Adversarial image tensor.
    """
    perturbation = epsilon * gradient.sign()
    adversarial_image = image_tensor - perturbation
    return torch.clamp(adversarial_image, min(image_range), max(image_range))


def generate_adversarial_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_category: str,
    categories: list[str],
    device: torch.device,
    epsilon: float,
    image_range: list[float, float],
) -> torch.Tensor:
    """
    Generate an adversarial image to misclassify the input image.

    Parameters
    ----------
        model: Pre-trained model.
        image_tensor: Preprocessed image tensor of shape
            (1, C, H, W).
        target_category: Desired misclassification target category.
        categories: List of model's categories.
        device: Target device (CPU or GPU).
        epsilon: Perturbation strength.

    Returns
    -------
        torch.Tensor: Adversarial image tensor.
    """
    # Ensure image tensor is in the correct dtype
    image_tensor = image_tensor.to(dtype=torch.float32)

    # Prepare tensor for target label
    target_tensor = convert_category_to_tensor(
        target_category, categories, device
    )
    # Ensure tensors are of dtype torch.long
    target_tensor = target_tensor.to(torch.long)

    # Compute loss gradient
    gradient = calculate_loss_gradient(model, image_tensor, target_tensor)

    # Apply perturbation
    return apply_perturbation(image_tensor, gradient, epsilon, image_range)
