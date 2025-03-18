from PIL import Image
import numpy as np
import tifffile

def rgb_reader(imgPath):
    img = Image.open(imgPath).convert("RGB")
    # img-->[H, W, C], dtype=np.uint8
    # Only img in this format can be normalized by torchvision.transforms.ToTensor()
    # For details: https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    img = np.array(img, dtype=np.uint8)
    return img

def tiff_reader(imgPath):
    img = tifffile.imread(imgPath)
    # tiff format is normally used for disparity or depth
    # so convert to float32 to avoid normalize while using torchvision.transforms.ToTensor()
    img = np.array(img, np.float32)
    return img

