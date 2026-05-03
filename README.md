# Chest X-Ray Classification: COVID-19, Pneumonia & Normal

A deep learning model that classifies chest X-ray images into three categories:
**COVID-19**, **Pneumonia**, and **Normal** — using transfer learning with ResNet50.

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 93.39% |
| F1 Score  | 93.40% |
| Recall    | 93.39% |
| Precision | 93.87% |

---

## Dataset

**COVID-19 Pneumonia Normal Chest X-Ray PA Dataset** (Kaggle)
- 6,902 images across 3 classes: `covid`, `normal`, `pneumonia`
- 80/20 train/validation split
- Image size: 256×256

---

## Model Architecture

The model uses **fine-tuned ResNet50** pretrained on ImageNet as a feature extractor,
with a custom classification head:

**Trainable params:** ~15.8M / **Total params:** ~23.6M

---

## Training Configuration

| Parameter       | Value                        |
|-----------------|------------------------------|
| Optimizer       | Adam (lr=1e-4)               |
| Loss            | SparseCategoricalCrossentropy|
| Epochs          | 60 (early stopping)          |
| Batch Size      | 32                           |
| Preprocessing   | ResNet50 `preprocess_input`  |

**Callbacks:**
- `EarlyStopping` — patience=10, monitors val_loss
- `ReduceLROnPlateau` — factor=0.5, min_lr=1e-7
- `ModelCheckpoint` — saves best model as `model.keras`

**Data Augmentation:**
- Random horizontal flip
- Random rotation (±10%)
- Random zoom (±15%)
- Random translation (±15%)

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn matplotlib seaborn pillow
```

### Training the Model

```python
# Load dataset
dataset = image_dataset_from_directory(
    PATH,
    seed=42,
    shuffle=True,
    image_size=(256, 256),
    batch_size=32
)

# Train
history = model.fit(
    train_ds,
    epochs=60,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
```

### Running Inference

```python
from tensorflow.keras.models import load_model

model = load_model('model.keras')

sample_predict(
    model=model,
    image_path='path/to/xray.jpeg'
)
```

---

## Project Structure

---

## Training Curves

The model achieved **~95% training accuracy** and **~95% validation accuracy**
by epoch 8, with early stopping triggered at epoch 18 due to no improvement
in validation loss.

---

## Evaluation

### Confusion Matrix

Evaluated on the validation set (43 batches, ~1,380 images):

Predicted
          covid  normal  pneumonia

*(See `Confusion Matrix.png` for the full heatmap)*

---

## Fine-Tuning Strategy

ResNet50's first 140 layers are frozen; only the final 40 layers
are fine-tuned. This approach:
- Preserves low-level ImageNet features
- Allows adaptation to medical imaging domain
- Prevents overfitting on a relatively small dataset

---

## Disclaimer

This model is intended for **research and educational purposes only**.
It should **not** be used for clinical diagnosis or medical decision-making
without proper validation by qualified medical professionals.

---

## References
- TensorFlow / Keras Documentation
- Dataset -> https://www.kaggle.com/code/monukushwahajarvis/covid-19-detection-resnet50-cnn-fine-tunning
  
The model uses **fine-tuned ResNet50**pretrained on ImageNet as a feature extractor,
with a custom classification head:
