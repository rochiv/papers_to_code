# 2D U-Net for Image Segmentation

## Overview
This repository contains an implementation of the 2D U-Net model for image segmentation, based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al. (2015).

## Architecture
The U-Net architecture consists of an encoder (contracting path) and a decoder (expansive path), which are connected by a bottleneck. The encoder captures context using convolutional layers and downsampling, while the decoder enables precise localization using upsampling and skip connections.

### Encoder
- Consists of repeated application of two 3x3 convolutions, each followed by a ReLU and a 2x2 max pooling operation with stride 2 for downsampling.

### Bottleneck
- Connects the encoder and decoder with two 3x3 convolutions followed by a ReLU.

### Decoder
- Each step in the decoder consists of an upsampling of the feature map followed by a 2x2 convolution that halves the number of feature channels, a concatenation with the corresponding feature map from the encoder, and two 3x3 convolutions each followed by a ReLU.

## Dataset
The model is designed to work with the ISBI Challenge: Segmentation of neuronal structures in EM stacks dataset. Ensure you have the dataset downloaded and organized into `images` and `masks` directories.

## Usage
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the training script with the dataset path.

## License
This project is licensed under the MIT License. 