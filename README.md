# Crop Disease Detection CNN

## Problem

Crop disease identification is often done manually, which is slow and inconsistent at scale. This project targets automated, image-based classification of crop health and disease for pepper, potato, and tomato using the PlantVillage dataset.

## Solution

Train a convolutional neural network (CNN) on labeled leaf images, then evaluate saved checkpoints. The training workflow is captured in a Jupyter notebook, and a separate notebook evaluates trained models.

## Architecture

- Training notebook: `Script/crop-disease-detection-cnn.ipynb`
	- Data loading and preprocessing
	- Augmentation pipeline
	- CNN model definition and training loop
	- Checkpoint saving (best model)
- Evaluation notebook: `Evalution/evaluate_model.ipynb`
	- Load trained model artifacts
	- Metrics and plots
- Artifacts: `model/` (saved model files)

Supporting docs:

- `Script/README.md`
- `Evalution/README.md`

## How To Run

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tqdm scikit-learn tensorflow
```

2. Open the training notebook:

- `Script/crop-disease-detection-cnn.ipynb`

3. Set the dataset path in the notebook:

- Update `data_dir` to your PlantVillage dataset folder.

4. Run the notebook top-to-bottom to train and save checkpoints.

5. Evaluate a saved model:

- Open `Evalution/evaluate_model.ipynb` and run it top-to-bottom.

## Notes

- Default image size is 128x128 with 3 channels.
- If you hit memory limits, reduce batch size or image size.
