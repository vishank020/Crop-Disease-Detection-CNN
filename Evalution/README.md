# Plant Disease Classification using CNN

A deep learning based plant disease classification system built using a Convolutional Neural Network (CNN) trained on the PlantVillage dataset. The project focuses on evaluating a pretrained TensorFlow/Keras .h5 model for multiclass crop disease detection and testing its real-world prediction capability.

### The implementation includes:
* Loading a pretrained CNN model
* Downloading dataset directly from Kaggle
* Evaluating model performance
* Generating classification metrics
* Visualizing confusion matrix
* Predicting disease from a single leaf image
* Saving the final evaluated model in .keras format

Project evaluation script reference:

## Dataset

### Dataset used: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

The dataset contains multiple plant leaf disease categories including:

* Pepper Bell Bacterial Spot
* Potato Early Blight
* Potato Late Blight
* Tomato Mosaic Virus
* Tomato Leaf Mold
* Healthy leaf classes
* Multiple bacterial and fungal infections

#### The dataset used in this implementation contains:

15 classes

## Objective

The objective of this project is to evaluate the performance of a pretrained CNN model on a multiclass plant disease dataset and analyze its prediction capability using standard deep learning evaluation metrics.

The workflow includes:

- Dataset acquisition from Kaggle
- Dataset preprocessing
- Loading pretrained model
- Model evaluation
- Accuracy analysis
- Confusion matrix generation
- Single image inference
- Final model export

### Technology Used

| Technology | Purpose |
| :--- | :--- |
| Python | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| Google Colab | Training & evaluation environment |
| Kaggle API | Dataset download |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| Seaborn | Heatmap visualization |
| Scikit-learn | Classification metrics |


### Project Structure
```tree
CNN-Crop-Disease/
│
├── model.h5
├── kaggle.json
├── leaf.jpg
├── evaluate_model.py
├── plant_disease_classifier.keras
├── README.md
│
├── dataset/
│   └── PlantVillage/
│       ├── Pepper__bell___Bacterial_spot/
│       ├── Pepper__bell___healthy/
│       ├── Potato___Early_blight/
│       ├── Potato___Late_blight/
│       ├── Tomato_Bacterial_spot/
│       └── ...
```
## Model Information

The pretrained model was loaded using TensorFlow/Keras:

```python
from tensorflow.keras.models import load_model

model = load_model("model.h5", compile=False)
```

The CNN expects image dimensions:

(128,128,3)

Meaning:

Height = 128
Width = 128
RGB channels = 3

## Kaggle API Configuration

The dataset was downloaded using Kaggle API authentication.

Configure Kaggle Credentials

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
Download Dataset

```bash
!kaggle datasets download -d emmarex/plantdisease
```

Extract Dataset

```bash
import zipfile
with zipfile.ZipFile("plantdisease.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")
```
### Dataset Preprocessing

Image preprocessing was performed using ImageDataGenerator.

```pyhton
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```
Validation split:

20%

This split was used as evaluation data.

### Data Generator
```bash
test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
```

### Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Loss function used:

- Categorical Cross Entropy

because the project performs multiclass classification.

## Model Evaluation

The model was evaluated using:

```python
loss, accuracy = model.evaluate(test_generator)
```

### Evaluation Results

| Metric | Value |
| :--- | :--- |
| Test Accuracy | 66.33% |
| Test Loss | 1.658 |

### Interpretation of Results

The CNN model achieved approximately:

66.33% accuracy

This indicates that the model correctly classifies around two-thirds of unseen plant disease images.

#### The performance suggests:

- The CNN is functional and capable of disease recognition
- The model generalizes moderately on validation data
- The architecture may still require optimization for production-level deployment

### Classification Report

Detailed classification metrics were generated using:
```bash
from sklearn.metrics import classification_report
```

#### Metrics include:

- Precision
- Recall
- F1-score
- Support

This helps analyze per-class performance and identify weak disease categories.

### Confusion Matrix

A confusion matrix heatmap was generated using:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
```
#### The confusion matrix helps identify:

- Misclassified diseases
- Similar disease confusion patterns
- Class imbalance issues


### Single Image Prediction

The model supports real-time prediction on custom leaf images.
```python

img = image.load_img(
    "leaf.jpg",
    target_size=(128,128)
)
```
#### Prediction pipeline:

- Load image
- Resize image
- Normalize pixel values
- Expand dimensions
- Run inference
- Return predicted disease label

### Model Saving

The evaluated model was saved using the modern Keras format:

```python
model.save("plant_disease_classifier.keras")
```

#### The .keras format is preferred over pickle because:

- Better TensorFlow compatibility
- Safer serialization
- More portable
- Preserves model metadata correctly
- Reduces version mismatch issues


## Future Improvements

#### Potential upgrades include:

- Transfer learning using:
    - EfficientNet
    - ResNet50
    - MobileNetV2
- Data augmentation
- Hyperparameter tuning
- Learning rate scheduling
- Early stopping
- Model quantization
- Streamlit deployment
- TensorFlow Lite conversion for mobile devices

## Conclusion

This project demonstrates a complete deep learning workflow for plant disease classification using CNNs and TensorFlow/Keras.

#### The implementation successfully:

- Evaluated a pretrained CNN model
- Processed multiclass plant disease data
- Generated performance metrics
- Performed real-world predictions
- Saved the final deployable model

The project also highlights practical debugging scenarios frequently encountered in deep learning pipelines, including shape mismatches, class inconsistencies, and preprocessing alignment issues.

### Reference

Evaluation script source: [Evaluation Script](Evalution\evaluate_model.ipynb)