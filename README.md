
# Image-to-Image Translation for the Cityscapes Dataset

This repository contains the code and resources used in our project to perform image-to-image translation on the Cityscapes dataset. The project structure is designed to facilitate understanding and interaction with the codebase.

## Project Structure

- **dataset folder**: This folder includes several Python scripts necessary to implement the `Cityscapes` class, which manages the input images. It also contains scripts to read the images and handle transformations for data augmentation.

- **gan folder**: Houses various Python files that implement the different generators and discriminators explored during the project. These files are crucial for understanding the modifications and tests carried out on the GAN architectures.

- **pretrained folder**: Contains the PyTorch state dictionaries used for the pretrained models.

- **requirements.txt**: Lists all the necessary packages along with their respective versions required to run the project. Ensure these dependencies are installed to avoid runtime issues.

- **.ipynb files**: Contains Jupyter notebooks for each experimental setup within the project. These notebooks detail the model testing and are key for replicating our results or for further experimentation.

## Setup and Installation

Before running the code, ensure that your environment meets all the dependencies listed in `requirements.txt`. You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

Explore the folders and notebooks for a comprehensive view of the experiments and methodologies employed in the image-to-image translation task.