# Adversarial Noise

A library to manipulate images to perform adversarial attacks based on pytorch classifiers.

## Description

Adversarial Noise manipulates images by adding adversarial noise. This noise is designed to trick an image classification model into misclassifying the altered image as a specified target class. This is achieved by calculating the gradient of the loss function for the input image and adding a perturbation that increases the loss. This type of modification is known as the adversarial Fast Gradient Sign Method (FGSM), described by [Goodfellow et al, 2015](https://arxiv.org/pdf/1412.6572).


## Installation

### Prerequisites
Install [Python 3.12](https://www.python.org/downloads/release/python-3127/) and [Poetry](https://python-poetry.org/docs/#installation).

Make poetry use Python 3.12 by running `poetry env use python3.12`.

### Poetry installation

After cloning the repository, install the dependencies and activate the virtual environment using poetry:

```
poetry install
poetry shell
````

Make sure to work with the poetry environment by selecting it as the interpreter. 

Dependencies are declared in `pyproject.toml`

## Running via the command line

Run the adversarial attack inputing the image path and the target category following the syntax:

``````
python -m adversarial_noise --image <IMAGE_PATH> --category <TARGET_CATEGORY>
``````

The user can provide other information, stored in a configuration file (`config.yml`)

### Parameters:

In the command line:
* `--image`: Name of the input image file (including path)
* `--category`: Target category, as a string
* `--config`: Path to the configuration file (default: config.yml) (optional)

In the config file (optional):
* `epsilon`: Strength of the modification. Ranges between [0, 1]. Default is 0.01.
* `output_path`: Path to save generated images. Default is "../../output_images/"
* `model_name`: Name of the pytorch classifying model. Default is "resnet50"
* `image_range`: Range for image clamping. Default is [0, 1]

## TO DO
* Iterate adversarial conversion until the image is successfully classified as the desired target.
* Assess performance with a wider range of images and target categories, considering training, validation, and testing datasets
* Do hyperparameter fine tuning. Different parameters, including epsilon, image size, image range for clamping, require exploration to opitmise performance. 
* Make other classifying models available e.g. other torchvision models.
* Explore different loss functions to generate noise (currently only used cross_entropy).
* Add safegurards to make sure the noise remains imperceptible.
* Add unit and end to end tests to increase testing coverage and robustness against edge cases.
* Include thorough project description e.g. details on rationale and overview.
* Package for its distribution.




