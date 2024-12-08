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
    image_tensor: torch.Tensor, gradient: torch.Tensor, epsilon: float
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
    return torch.clamp(adversarial_image, -0.5, 0.5)

