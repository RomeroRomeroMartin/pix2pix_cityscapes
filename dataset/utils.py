
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
        

def visualize_samples(dataset, num_samples=5):
    plt.figure(figsize=(10, 2 * num_samples))
    
    for i in range(num_samples):
        # Randomly select an index to visualize
        idx = torch.randint(len(dataset), size=(1,)).item()
        
        # Fetch the images from the dataset
        imgA, imgB = dataset[idx]
        
        # Plotting the images
        ax = plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(imgA.permute(1, 2, 0))  # Adjusting the channel dimension for matplotlib
        ax.set_title('Input Image')
        ax.axis('off')
        
        ax = plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(imgB.permute(1, 2, 0))  # Adjusting the channel dimension for matplotlib
        ax.set_title('Target Image')
        ax.axis('off')
    
    plt.show()



def visualize_normalization(data_loader, mean, std, num_images=1):
    # Adjust the subplot to include output images
    fig, axs = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))  # 4 columns: Input (before/after), Output (before/after)
    
    # Make axs always 2D array even if num_images is 1
    if num_images == 1:
        axs = [axs]  # Wrap it in a list so indexing remains consistent
    
    for i, ((imgA, imgB)) in enumerate(data_loader):
        if i >= num_images:  # Only display the specified number of images
            break
        
        for j in range(num_images):
            # Normalize and un-normalize for visualization for both imgA and imgB
            imgA_unnorm = imgA[0] * std[:, None, None] + mean[:, None, None]  # Un-normalize the first input image
            imgA_unnorm = imgA_unnorm.clamp(0, 1)  # Clamp to ensure the image values are valid
            
            imgB_unnorm = imgB[0] * std[:, None, None] + mean[:, None, None]  # Un-normalize the first output image
            imgB_unnorm = imgB_unnorm.clamp(0, 1)

            # Subplots for the un-normalized input and output images
            axs[j][0].imshow(imgA_unnorm.permute(1, 2, 0))
            axs[j][0].set_title('Input Before Normalization')
            axs[j][0].axis('off')
            
            axs[j][1].imshow(imgA[0].permute(1, 2, 0))
            axs[j][1].set_title('Input After Normalization')
            axs[j][1].axis('off')

            # Subplots for the un-normalized input and output images
            axs[j][2].imshow(imgB_unnorm.permute(1, 2, 0))
            axs[j][2].set_title('Output Before Normalization')
            axs[j][2].axis('off')
            
            axs[j][3].imshow(imgB[0].permute(1, 2, 0))
            axs[j][3].set_title('Output After Normalization')
            axs[j][3].axis('off')
        
        plt.show()
        return  # Exit after the first batch to prevent showing more batches


import numpy as np

def convert2_0_1(imagen):
    """
    Convierte una imagen normalizada en el rango de -1 a 1 a un rango de 0 a 255.

    Args:
        imagen: Tensor o array de NumPy que representa la imagen en el rango de -1 a 1.

    Returns:
        imagen_0_255: Array de NumPy que representa la imagen en el rango de 0 a 255.
    """
    # Primero convertimos la imagen al rango de 0 a 1
    imagen_0_1 = (imagen + 1) / 2
    
    # Luego, convertimos al rango de 0 a 255
    #imagen_0_255 = imagen_0_1 * 255
    
    # Convertimos los valores a enteros
    #imagen_0_255 = imagen_0_255.to(torch.uint8)
    
    return imagen_0_1