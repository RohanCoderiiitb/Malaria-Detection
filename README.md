#  Malaria Detection using Deep Learning and TensorFlow Lite

This project implements a deep learning pipeline to detect malaria-infected blood cells using TensorFlow, leveraging transfer learning with MobileNetV2 and deploying optimized models via TensorFlow Lite.

## 📂 Overview

* 📅 Dataset: [Malaria Dataset](https://www.tensorflow.org/datasets/catalog/malaria) from TensorFlow Datasets
* 🧠 Model: Pre-trained **MobileNetV2** from TensorFlow Hub
* ⚙️ Training: Conducted on resized and normalized cell images
* 📱 Deployment: Model converted to TFLite with multiple quantization strategies
* 📊 Evaluation: Accuracy tested on the original and quantized models

---

## 🚀 Features

* ✅ Image resizing and normalization
* ✅ MobileNetV2 feature extraction
* ✅ Classification using a fully connected softmax layer
* ✅ Model training and validation
* ✅ Saving and converting model to TFLite format
* ✅ Four types of model quantization:

  * Default
  * Optimize for latency
  * Optimize for size
  * Experimental sparsity
* ✅ Evaluation of each quantized model's accuracy

---

## 🛠️ Requirements

* Python 3.7+
* TensorFlow 2.x
* TensorFlow Hub
* TensorFlow Datasets
* NumPy
* tqdm

Install dependencies:

```bash
pip install tensorflow tensorflow-hub tensorflow-datasets numpy tqdm
```

---

## 🧪 Training the Model

```bash
python MalariaDetection.py
```

The script will:

1. Download and preprocess the dataset.
2. Train a MobileNetV2-based model.
3. Convert it to TFLite.
4. Apply four types of quantization.
5. Evaluate the accuracy of all models on the test set.

---

## 📁 Output

* `SavedModels/Model1/` — TensorFlow SavedModel
* `/tmp/model1.tflite` — Baseline TFLite model
* `/tmp/tfmodel1.tflite` to `/tmp/tfmodel4.tflite` — Quantized variants

---

## 📌 Notes

* The model is not fine-tuned; only the top classifier layer is trained.
* Intended for deployment on mobile or embedded devices.
* You can modify the number of training epochs or batch size as needed.

---

## 🚧 Project Status

This project is **ongoing**.

Planned enhancements include:

* Training a **custom Convolutional Neural Network (CNN)** model from scratch (instead of using a pre-trained MobileNetV2)
* Applying all four quantization techniques to the custom model
* Converting and evaluating the custom model in TensorFlow Lite format for efficient deployment

