import torch
import cv2
import argparse
import numpy as np
from skimage.color import lab2rgb, rgb2gray
import os

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
    l_channel = gray_tensor[0].cpu().numpy() * 100
    ab_channel = ab_pred[0].cpu().numpy() * 255 - 128
    lab = np.concatenate((l_channel[None, :, :], ab_channel), axis=0).transpose(1, 2, 0)
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
    
    colorize_image(args.model, args.image, args.output)
