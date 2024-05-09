'''import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Generador basado en DeepLabV3 preentrenado
class Generator(nn.Module):
    def __init__(self, out_channels=3):
        super(Generator, self).__init__()

        # Cargar DeepLabV3 preentrenado
        self.base_model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # Ajustar la última capa de salida
        #self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)  # Cambia los canales de salida según sea necesario
        #self.final_conv = nn.ConvTranspose2d(2*64, 3, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(2048, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        print(f'Input to Generator: {x.shape}')

        # Adelante a través del modelo base (backbone) de DeepLabV3
        features = self.base_model(x)['out']
        print(f'Output from DeepLabV3: {features.shape}')

        # Añadir la capa final de convolución para generar las imágenes
        out = self.final_conv(features)
        print(f'Output from final convolution: {out.shape}')

        # Ajusta el tamaño de la imagen de salida si es necesario
        # Por ejemplo, para obtener imágenes de tamaño de entrada original
        if out.size()[2] != x.size()[2] or out.size()[3] != x.size()[3]:
            out = F.interpolate(out, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
            print(f'Output after interpolation: {out.shape}')


        return x'''
    
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Generador basado en DeepLabV3 preentrenado
class Generator(nn.Module):
    def __init__(self, out_channels=3):
        super(Generator, self).__init__()

        # Cargar DeepLabV3 preentrenado
        self.deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # Usar la capa de características (backbone) de DeepLabV3
        self.base_model = self.deeplabv3.backbone

        # Añadir una capa final para la generación de imágenes
        # Asegúrate de que el número de canales de entrada coincida con la salida de DeepLabV3
        self.final_conv = nn.Conv2d(2048, out_channels, kernel_size=3, padding=1)

    def forward(self, x):


        # Adelante a través del modelo base (backbone) de DeepLabV3
        features = self.base_model(x)
        
        # Salida de DeepLabV3 es un diccionario; selecciona la clave 'out'
        if isinstance(features, dict):
            features = features.get('out')

        # Añadir la capa final de convolución para generar las imágenes
        out = self.final_conv(features)

        # Ajusta el tamaño de la imagen de salida si es necesario
        # Por ejemplo, para obtener imágenes de tamaño de entrada original
        if out.size()[2] != x.size()[2] or out.size()[3] != x.size()[3]:
            out = F.interpolate(out, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)

        return out


