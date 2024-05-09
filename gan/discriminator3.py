import torch
import torch.nn as nn
from torchvision import models

# Discriminador basado en ResNet
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Usar ResNet18 preentrenado
        resnet = models.resnet18(pretrained=True)

        # Ajustar la capa de entrada para aceptar 6 canales (3 de real/fake + 3 de cond)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Usar las capas de ResNet después de la primera capa
        self.resnet_layers = nn.Sequential(
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Calcula el número de características aplanadas correctamente
        # Puedes usar el cálculo como resnet.layer4[-1].conv2.out_channels * 8 * 8
        num_ftrs = 512 * 8 * 8

        # Ajustar la capa final de salida
        self.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond=None):
        # Combinar img y cond si cond no es None
        if cond is not None:
            img = torch.cat([img, cond], dim=1)  # Concatenar a lo largo de la dimensión de los canales
        
        # Pasar a través de la primera capa de convolución ajustada
        img = self.conv1(img)
        
        # Adelante a través de las capas restantes de ResNet
        img = self.resnet_layers(img)
        
        # Aplanar el tensor antes de pasarlo a la capa completamente conectada
        img = img.view(img.size(0), -1)

        # Pasar a través de la capa completamente conectada
        img = self.fc(img)
        
        # Aplicar activación sigmoid
        img = self.sigmoid(img)
        
        return img
