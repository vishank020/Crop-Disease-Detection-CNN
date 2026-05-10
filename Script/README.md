# Crop Disease Detection CNN (Notebook)

Train a DenseNet-based CNN on the PlantVillage dataset for multi-class crop disease classification.

## What This Notebook Does

- Loads images from the PlantVillage dataset.
- Builds a DenseNet121-based CNN with custom classification head.
- Trains with data augmentation and checkpoints the best model.
- Visualizes class distribution and sample images.

## Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - opencv-python
  - tqdm
  - scikit-learn
  - tensorflow

You can install these with:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tqdm scikit-learn tensorflow
```

## Dataset

The notebook expects the PlantVillage dataset structured by class folders. Update `data_dir` in the notebook to point to your local dataset path. Example structure:

```
PlantVillage/
  Pepper__bell___Bacterial_spot/
  Pepper__bell___healthy/
  Potato___Early_blight/
  Potato___Late_blight/
  Potato___healthy/
  Tomato_Bacterial_spot/
  Tomato_Early_blight/
  Tomato_Late_blight/
  Tomato_Leaf_Mold/
  Tomato_Septoria_leaf_spot/
  Tomato_Spider_mites_Two_spotted_spider_mite/
  Tomato__Target_Spot/
  Tomato__Tomato_YellowLeaf__Curl_Virus/
  Tomato__Tomato_mosaic_virus/
  Tomato_healthy/
```

This maps to 15 classes defined in the notebook's `disease_types` list.

## Notebook Pipeline

1. Load libraries and configure constants.
2. Build a dataframe of image paths and labels.
3. Visualize class distribution and sample images.
4. Load and resize images, then normalize to `[0, 1]`.
5. One-hot encode labels and create a stratified train/validation split.
6. Build and train a DenseNet121-based model with augmentation.

## Key Configuration (Defaults)

- Image size: 128x128
- Channels: 3
- Batch size: 16
- Epochs: 25
- Seed: 42

## Model Details

- Backbone: DenseNet121 (`include_top=False`, `weights=None`)
- Input adapter: 3x3 Conv2D to match channel expectations
- Head: GlobalAveragePooling2D + BatchNorm + Dropout + Dense(256) + BatchNorm + Dropout + Dense(15, softmax)
- Optimizer: Adam (learning rate 0.002, epsilon 0.1)
- Loss: categorical crossentropy

## Training Details

- Train/validation split: 80/20 with stratification
- Augmentation: rotation, shifts, zoom, horizontal/vertical flips
- Callbacks: ReduceLROnPlateau and ModelCheckpoint

## How To Run

1. Open the notebook: `crop-disease-detection-cnn.ipynb`.
2. Set the correct `data_dir` path in the dataset cell.
3. Run cells top-to-bottom.

## Outputs

- Training history plots in the notebook.
- Best model checkpoint saved as `model.h5` in the current working directory.

## Repository Artifacts

- Pretrained weights (if present): `model/model.h5`
- Alternate format (if present): `model/plant_disease_classifier.keras`

## Notes

- The notebook uses a fixed image size of 128x128 and 15 classes.
- If you want to save the model elsewhere, update the `ModelCheckpoint` path.
 - If you hit memory errors, reduce `BATCH_SIZE` or image size.
