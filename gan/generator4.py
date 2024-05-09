import torch
import torch.nn as nn

# Definición de un bloque de ResNet
class ResnetBlock(nn.Module):
    """Define un bloque de ResNet con conexiones de salto."""
    
    def __init__(self, dim, norm_layer, use_dropout=False):
        """Constructor del bloque de ResNet."""
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout)

    def build_conv_block(self, dim, norm_layer, use_dropout):
        """Construye el bloque de convolución."""
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Propagación hacia adelante con conexiones de salto."""
        return x + self.conv_block(x)

# Definición de la arquitectura del generador
class Generator(nn.Module):
    """Generador basado en ResNet."""
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9):
        """Constructor del generador basado en ResNet."""
        super(Generator, self).__init__()

        # Capa inicial
        self.initial_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # Bloques de bajada
        self.down_layers = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # Bloques de ResNet
        resnet_blocks = []
        for _ in range(n_blocks):
            resnet_blocks.append(ResnetBlock(ngf * 4, norm_layer, use_dropout))
        self.resnet_layers = nn.Sequential(*resnet_blocks)

        # Bloques de subida
        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        """Propagación hacia adelante."""
        x = self.initial_layers(x)
        x = self.down_layers(x)
        x = self.resnet_layers(x)
        x = self.up_layers(x)
        x = self.output_layer(x)
        return x
