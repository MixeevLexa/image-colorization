import torch
import cv2
import argparse
import numpy as np
from skimage.color import lab2rgb, rgb2gray
from PIL import Image
import os

def colorize_image(model_path: str, image_path: str, output_path: str = "colorized.jpg"):
    """
    Colorize a grayscale image using the trained model.
    
    Args:
        model_path (str): Path to the trained model weights (.pth file).
        image_path (str): Path to the input image.
        output_path (str): Path where the colorized image will be saved.
    
    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    img = cv2.resize(img, (256, 256))
    gray = rgb2gray(img)
    gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        ab_pred = model(gray_tensor)

    # Reconstruct LAB image and convert to RGB
    l_channel = gray_tensor[0].cpu().numpy() * 100
    ab_channel = ab_pred[0].cpu().numpy() * 255 - 128
    lab = np.concatenate((l_channel[None, :, :], ab_channel), axis=0).transpose(1, 2, 0)
    rgb = (lab2rgb(lab) * 255).astype(np.uint8)

    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Colorized image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize a grayscale image using trained CNN model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input grayscale image")
    parser.add_argument("--model", type=str, default="models/model.pth", help="Path to the trained model weights")
    parser.add_argument("--output", type=str, default="colorized.jpg", help="Output path for the colorized image")
    
    args = parser.parse_args()
    
    colorize_image(args.model, args.image, args.output)
