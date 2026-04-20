import torch
import torch.nn as nn
from torchvision import models
import cv2
import argparse
import numpy as np
from skimage.color import lab2rgb, rgb2gray
import os

class ColorizationNet(nn.Module):
    """
    CNN for image colorization.
    Encoder: first 6 layers of ResNet34 (adapted for 1-channel grayscale input).
    Decoder: series of convolutions + upsampling to predict AB channels in LAB color space.
    """
    def __init__(self):
        super(ColorizationNet, self).__init__()
        resnet = models.resnet34(weights=None)
        # Adapt first convolution for grayscale (1 channel)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1, keepdim=True))

        self.encoder = nn.Sequential(*list(resnet.children())[:6])

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        """Forward pass: grayscale -> predicted AB channels."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def colorize_image(image_path: str, model_path: str = "model.pth", output_path: str = "colorized.jpg"):
    """
    Colorize a grayscale or color image using the trained CNN model.
    
    Args:
        model_path (str): Path to the saved model weights (.pth)
        image_path (str): Path to the input image
        output_path (str): Path to save the colorized image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    original_size = (img.shape[1], img.shape[0]) 
    img_resized = cv2.resize(img, (256, 256))
    gray = rgb2gray(img_resized)
    gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        ab_pred = model(gray_tensor)

    # Reconstruct LAB → RGB
    l_channel = gray_tensor[0, 0].cpu().numpy() * 100
    ab_channel = ab_pred[0].cpu().numpy() * 255 - 128
    lab = np.zeros((256, 256, 3), dtype=np.float32)
    lab[:, :, 0] = l_channel
    lab[:, :, 1] = ab_channel[0, :, :]
    lab[:, :, 2] = ab_channel[1, :, :]
    rgb = (lab2rgb(lab) * 255).astype(np.uint8)

    # Resize colorized image back to original size
    rgb_resized = cv2.resize(rgb, original_size, interpolation=cv2.INTER_LINEAR)

    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
    print(f"Colorized image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize image using trained colorization model")
    parser.add_argument("--image",  type=str, required=True,  help="Path to input image")
    parser.add_argument("--model",  type=str, default="model.pth", help="Path to model weights")
    parser.add_argument("--output", type=str, default="colorized.jpg",    help="Output image path")
    
    args = parser.parse_args()
    
    colorize_image(args.image, args.model, args.output)
