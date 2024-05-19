# Flower Classification using Vision Transformer (ViT)

This project is an implementation of the Vision Transformer (ViT) model for the task of flower classification. The model is trained to classify three types of flowers: hibiscus, rose, and sunflower.
Link to dataset: https://universe.roboflow.com/kani-tamil/flower-fgdvh/dataset/2/images

## Dataset

The dataset used for training and evaluation is a subset of the popular flowers dataset. It consists of images of hibiscus, rose, and sunflower flowers. The dataset is split into training,validation and testing sets.

## Model Architecture

The Vision Transformer (ViT) model is a transformer-based architecture designed for image classification tasks. It treats an image as a sequence of patches and processes them through a standard transformer encoder, enabling the model to capture long-range dependencies and relationships between different parts of the image.

## Implementation Details

- **Framework**: PyTorch
- **Model**: Vision Transformer (ViT)
- **Number of Classes**: 3 (hibiscus, rose, sunflower)
- **Training Epochs**: 5
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

## Getting Started

1. Clone the repository:
  
2. Install the required dependencies:
   `pip install matplotlib torch torchvision`
3. Update the `train_dir` and `val_dir` paths in the code to point to your training and validation data directories, respectively.

4. Run the training script:
    `Vit_from_scratch.ipynb`
The training progress and validation metrics will be printed to the console.

## Results

After training for 5 epochs, the model achieved the following performance on the validation set:

- Validation Accuracy: 71%
- Validation Loss: 0.648

## Future Work

- Explore different ViT architectures and hyperparameters for improved performance.
- Extend the model to classify more flower varieties.
- Implement techniques like data augmentation and transfer learning for better generalization.

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT paper)
- [Vision Transformers, Explained](https://towardsdatascience.com/vision-transformers-explained-a9d07147e4c8)

