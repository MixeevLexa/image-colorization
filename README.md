# Image Colorization

## Overview
The project implements an encoder-decoder architecture that takes a single-channel grayscale image as input and outputs the corresponding A and B channels in the LAB color space. The final colorized image is obtained by combining the original lightness channel with the predicted chrominance.

## Model Architecture

### Color Space Conversion
Images are processed in the **LAB** color space:
- **L** channel (Lightness) serves as the input to the network, representing the grayscale image.
- **A** and **B** channels (chrominance) are predicted by the model.
- At inference, the predicted AB channels are merged with the original L channel and converted back to RGB.

### Network Design
- **Encoder**:  
  Utilizes the first six layers of **ResNet34** as a feature extractor. The initial convolutional layer is modified to accept single-channel input while preserving the pretrained architecture's feature extraction capabilities.

- **Decoder**:  
  A lightweight upsampling decoder composed of convolutional layers, Batch Normalization, and ReLU activations. Progressive ×2 upsampling is applied to restore the original spatial resolution. The final output layer produces 2 channels corresponding to the A and B components.

- **Input / Output**:
  - Input: Grayscale tensor of shape `(batch_size, 1, 224, 224)`
  - Output: AB channels tensor of shape `(batch_size, 2, 224, 224)`

- **Loss Function**: Mean Squared Error (MSE) between predicted and ground-truth AB channels.

## Dataset
The model is trained on the [Landscape Image Colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization) dataset.

## Usage

Clone this repository
```bash
git clone https://github.com/MixeevLexa/image-colorization.git
cd image-colorization
```

Install the required modules
```bash
pip install -r requirements.txt
```

Run `colorize.py` with grayscale image that needs to color

```bash
python colorize.py --image path/to/your_grayscale_image.jpg
```

## Results
