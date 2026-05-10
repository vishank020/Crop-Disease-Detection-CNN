# Crop Disease Detection CNN

A CNN-based project for classifying crop diseases using the PlantVillage dataset. The main training workflow is captured in a Jupyter notebook, and the repository includes trained model artifacts and a separate evaluation notebook.

## Project Structure

- `Script/` - Training notebook and documentation.
- `Evalution/` - Model evaluation notebook and documentation.
- `model/` - Saved model artifacts.

## Notebooks

- `Script/crop-disease-detection-cnn.ipynb` - End-to-end training pipeline with data loading, augmentation, model building, and training.
- `Evalution/evaluate_model.ipynb` - Evaluation workflow for trained models.

For notebook-specific details, see:

- `Script/README.md`
- `Evalution/README.md`

## Dataset

This project uses the PlantVillage dataset organized by class folders. Update the `data_dir` variable in the training notebook to point to your local dataset path. The dataset includes 15 disease/healthy classes across pepper, potato, and tomato.

## Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Core packages: numpy, pandas, matplotlib, seaborn, opencv-python, tqdm, scikit-learn, tensorflow

Install example:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tqdm scikit-learn tensorflow
```

## Outputs

- Training history plots generated in the notebook.
- Best model checkpoint saved as `model.h5` during training.
- Optional saved artifacts in `model/`.

## Quick Start

1. Open `Script/crop-disease-detection-cnn.ipynb`.
2. Set the `data_dir` path to your dataset location.
3. Run the notebook top-to-bottom.
4. Use `Evalution/evaluate_model.ipynb` to evaluate saved models.

## Notes

- Default image size is 128x128 with 3 channels.
- If you encounter memory issues, reduce the batch size or image size.
