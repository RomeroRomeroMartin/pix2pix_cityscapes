import PIL
import torch
import numpy as np
from PIL import Image

class Transformer(object):
    """"Transform"""
    def __init__(self,):
        pass
    def __call__(self, imgA, imgB=None):
        pass

class Compose(Transformer):
    """Compose transforms"""
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms=transforms
        
    def __call__(self, imgA, imgB=None):
        if imgB is None:
            for transform in self.transforms:
                imgA = transform(imgA, imgB)
            return imgA
        for transform in self.transforms:
            imgA, imgB = transform(imgA, imgB)
        return imgA, imgB
    
class Resize(Transformer):
    """Resize imageA and imageB"""
    def __init__(self, size=(256, 256)):
        """
        :param: size (default: tuple=(256, 256)) - target size
        """
        super().__init__()
        self.size=size
        
    def __call__(self, imgA, imgB=None):
        imgA = imgA.resize(self.size)
        if imgB is None:
            return imgA
        imgB = imgB.resize(self.size)
        return imgA, imgB
    
class CenterCrop(Transformer):
    """CenterCrop imageA and imageB"""
    def __init__(self, size=(256, 256), p=0.5):
        """
        :param: size (default: tuple=(256, 256)) - target size
        """
        super().__init__()
        self.size=size
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        W, H = imgA.size
        cW, cH = W//2, H//2
        fW, fH = self.size[0]//2, self.size[1]//2
        if np.random.uniform() < self.p:
            imgA = imgA.crop((cW-fW, cH-fH, cW+fW, cH+fH))
            if imgB is not None:
                imgB = imgB.crop((cW-fW, cH-fH, cW+fW, cH+fH))
        else:
            imgA = imgA.resize(self.size)
            if imgB is not None:
                imgB = imgB.resize(self.size)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class Rotate(Transformer):
    """Rotate imageA and imageB"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of rotation
        """
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = imgA.rotate(180)
            if imgB is not None:
                imgB = imgB.rotate(180)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class HorizontalFlip(Transformer):
    """Horizontal flip of imageA and imageB"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of horizontal flip
        """
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = imgA.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            if imgB is not None:
                imgB = imgB.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class VerticalFlip(Transformer):
    """Vertical flip of imageA and imageB"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of vertical flip
        """
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = imgA.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            if imgB is not None:
                imgB = imgB.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class ToTensor(Transformer):
    """Convert imageA and imageB to torch.tensor"""
    def __init__(self,):
        super().__init__()
    
    def __call__(self, imgA, imgB=None):
        imgA = np.array(imgA)/255.
        imgA = torch.from_numpy(imgA).float().permute(2, 0, 1)
        if imgB is None:
            return imgA
        imgB = np.array(imgB)/255.
        imgB = torch.from_numpy(imgB).float().permute(2, 0, 1)
        return imgA, imgB
    
class ToImage(Transformer):
    """Convert imageA and imageB tensors to PIL.Image"""
    def __init__(self,):
        super().__init__()
    
    def __call__(self, imgA, imgB=None):
        imgA = imgA.permute(1,2,0).numpy()
        imgA = Image.fromarray(np.uint8(imgA*255))
        if imgB is None:
            return imgA
        imgB = imgB.permute(1,2,0).numpy()
        imgB = Image.fromarray(np.uint8(imgB*255))
        return imgA, imgB
    
class Normalize(Transformer):
    """Normalize imageA and imageB"""
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        :param: mean (default: list=[0.485, 0.456, 0.406]) - list of means for each image channel 
        :param: std (default: list=[0.229, 0.224, 0.225]) - list of stds for each image channel
        """
        super().__init__()
        self.mean=mean
        self.std=std
        
    def __call__(self, imgA, imgB=None):
        
        if (self.mean is not None) and (self.std is not None):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            imgA = (imgA - mean)/std
            if imgB is not None:
                imgB = (imgB - mean)/std
            
        if imgB is None:
            return imgA
        return imgA, imgB
    
class DeNormalize(Transformer):
    """DeNormalize imageA and imageB"""
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        :param: mean (default: list=[0.485, 0.456, 0.406]) - list of means for each image channel 
        :param: std (default: list=[0.229, 0.224, 0.225]) - list of stds for each image channel
        """
        super().__init__()
        self.mean=mean
        self.std=std
        
    def __call__(self, imgA, imgB=None):
        
        if (self.mean is not None) and (self.std is not None):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            imgA = imgA*std + mean
            imgA = torch.clip(imgA, 0., 1.)
            if imgB is not None:
                imgB = imgB*std + mean
                imgB = torch.clip(imgB, 0., 1.)
            
        if imgB is None:
            return imgA
        return imgA, imgB
from PIL import ImageEnhance, Image

class CustomColorJitter(Transformer):
    """
    CustomColorJitter: Ajusta el brillo, contraste, saturación y tono de una imagen.

    Args:
        brightness (float): Cantidad de ajuste de brillo. Valores mayores de 0 indican mayor brillo, valores menores indican menor brillo.
        contrast (float): Cantidad de ajuste de contraste. Valores mayores de 0 indican mayor contraste, valores menores indican menor contraste.
        saturation (float): Cantidad de ajuste de saturación. Valores mayores de 0 indican mayor saturación, valores menores indican menor saturación.
        hue (float): Cantidad de ajuste de tono. Valores mayores o menores de 0 indican un cambio en el tono de la imagen.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, imgA, imgB=None):
        # Aplica el ajuste de color a imgA
        imgA = self.apply_color_jitter(imgA)
        # Si imgB no es None, aplica los ajustes a imgB
        if imgB is not None:
            imgB = self.apply_color_jitter(imgB)
        # Retorna imgA e imgB transformadas en formato PIL Image
        return imgA, imgB
    
    def apply_color_jitter(self, img):
        # Aplica el ajuste de brillo
        enhancer = PIL.ImageEnhance.Brightness(img)
        brightness_factor = 1 + self.brightness * (np.random.uniform() * 2 - 1)
        img = enhancer.enhance(brightness_factor)
        
        # Aplica el ajuste de contraste
        enhancer = PIL.ImageEnhance.Contrast(img)
        contrast_factor = 1 + self.contrast * (np.random.uniform() * 2 - 1)
        img = enhancer.enhance(contrast_factor)
        
        # Aplica el ajuste de saturación
        enhancer = PIL.ImageEnhance.Color(img)
        saturation_factor = 1 + self.saturation * (np.random.uniform() * 2 - 1)
        img = enhancer.enhance(saturation_factor)
        
        # Aplica el ajuste de tono (hue)
        # Utiliza el método .convert() para cambiar la imagen a modo HSV, ajusta el tono, y luego convierte de vuelta a RGB
        if self.hue != 0:
            img = img.convert('HSV')
            # Obtén los datos de la imagen
            data = np.array(img)
            # Aplica el ajuste de tono a la imagen
            data[:, :, 0] = (data[:, :, 0].astype(np.float32) + self.hue * 255) % 255
            # Convierte los datos modificados de vuelta a una imagen HSV
            img = Image.fromarray(data, mode='HSV')
            # Convierte la imagen de HSV a RGB
            img = img.convert('RGB')
        
        return img

    
__all__ = ['Transformer', 'Compose', 'Resize', 'CenterCrop', 'Rotate', 'HorizontalFlip',
           'VerticalFlip', 'ToTensor', 'ToImage', 'Normalize', 'DeNormalize','CustomColorJitter',]