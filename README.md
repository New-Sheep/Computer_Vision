
# 🏥 Multi-Object Detection of Surgical Instruments using YOLOv8

This repository contains the code and configuration files used to train a **YOLOv8-Small** model for detecting and classifying surgical instruments in cluttered tray environments.

---

## 🎯 Detected Classes

The model detects the following surgical instruments:

- Bistoury  
- Dissection Forceps  
- Straight Scissors  
- Curved Scissors  

---

## 📁 Project Structure

```
├── create_yolo_dataset.py   # Converts original JSON annotations to YOLO format (.txt)
├── verify_txt.py            # Visual verification of YOLO bounding boxes
├── eda_analysis.py          # Generates class distribution & bounding box charts
├── train_yolo.py            # Training script with augmentation & tuning
├── generate_submission.py   # Runs inference and generates submission JSON
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

> **Note:** Ensure PyTorch is installed with CUDA support if training on a GPU.

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 How to Run the Code

### 1️⃣ Data Preparation

Ensure the folder:

```
yolo_surgical_dataset/
```

exists in the root directory and contains properly structured:

```
train/
val/
```

splits in YOLO format.

---

### 2️⃣ Training the Model

Start 50-epoch training with:

```bash
python train_yolo.py
```

This will generate:

```
runs/detect/CE888_Runs/
```

Inside this directory you will find:

- Loss curves  
- Confusion matrices  
- `best.pt` (best trained model weights)

---

### 3️⃣ Generating Test Predictions

Generate the final JSON submission file:

```bash
python generate_submission.py
```

This script will:

1. Load `best.pt`
2. Run inference on test images
3. Output a JSON file formatted according to `sample_submission_test.json`

---

## 🧠 Training Details

The training pipeline includes:

- YOLOv8-Small architecture  
- Mosaic augmentation  
- Rotational augmentation  
- HSV color augmentation  
- Custom hyperparameter tuning  

---

## 📊 Evaluation & Validation

Visualize bounding boxes:

```bash
python verify_txt.py
```

Generate EDA plots:

```bash
python eda_analysis.py
```

---

## 🏁 Output

After successful training, the best-performing model will be saved as:

```
runs/detect/CE888_Runs/weights/best.pt
```

This file is used for inference and submission generation.

---

## 📌 Notes

- Ensure dataset annotations are correctly normalized for YOLO format.
- Double-check class index ordering before training.
- GPU training is strongly recommended.

---

## 👨‍⚕️ Author

Developed for surgical instrument detection in cluttered operating tray environments using YOLOv8.
