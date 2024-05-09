import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """PatchGAN discriminador para CycleGAN."""
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Constructor del discriminador PatchGAN.
        - input_nc: Número de canales de entrada.
        - ndf: Número de filtros en la capa inicial.
        - n_layers: Número de capas de convolución.
        - norm_layer: Capa de normalización.
        """
        super(Discriminator, self).__init__()
        # Capa inicial
        kw = 4  # tamaño del kernel
        padw = 1  # padding del kernel
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        
        # Capas intermedias
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Capa final
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Capa de salida
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Propagación hacia adelante."""
        return self.model(input)
