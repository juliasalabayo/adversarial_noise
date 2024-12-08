import logging

import torch

from adversarial_noise.utils import convert_category_to_tensor

logger = logging.getLogger("adversarial_noise")


def calculate_loss_gradient(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the gradient of the loss of the input image in comparison to the
    target label.

    Parameters
    ----------
        model: Pre-trained model.
        image_tensor: Preprocessed image tensor.
        target_tensor: Target category tensor.

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


def compare_confidences(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_category: str,
    categories: list[str],
    adversarial_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compare the confidence for the target category in the original and
    adversarial images.

    Parameters
    ----------
        model: Pre-trained model.
        image_tensor: Preprocessed image tensor of shape
            (1, C, H, W).
        target_category: Desired misclassification target category.
        categories: List of model's categories.
        adversarial_tensor: Tensor of the image with adversarial noise.

    Returns
    -------
        torch.Tensor: Adversarial image tensor.
    """
    with torch.no_grad():
        # Get model's predictions for the original image
        original_pred = model(image_tensor)
        original_pred_prob = torch.nn.functional.softmax(original_pred, dim=1)
        original_confidence = original_pred_prob[
            0, categories.index(target_category)
        ].item()

    # Get model's predictions for the adversarial image
    adversarial_pred = model(adversarial_tensor)
    adversarial_pred_prob = torch.nn.functional.softmax(
        adversarial_pred, dim=1
    )
    adversarial_confidence = adversarial_pred_prob[
        0, categories.index(target_category)
    ].item()

    # Log the confidences
    logger.info(
        f"Original confidence for target category '{target_category}': "
        f"{original_confidence:.4f}"
    )
    logger.info(
        f"Adversarial confidence for target category '{target_category}'"
        f": {adversarial_confidence:.4f}"
    )
    logger.info(
        f"Adversarial category predicted as "
        f"'{categories[adversarial_pred.argmax()]}'"
    )

    # Compare confidences
    if adversarial_confidence < original_confidence:
        logger.info(
            "The adversarial image reduced target category confidence."
        )
    elif adversarial_confidence > original_confidence:
        logger.info(
            "The adversarial image increased target category confidence."
        )
    else:
        logger.info("The confidence for the target category remains the same.")
    return adversarial_pred
