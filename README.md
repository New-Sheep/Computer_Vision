Multi-Object Detection of Surgical Instruments using YOLOv8
This repository contains the code and configuration files used to train a YOLOv8-Small model for detecting and classifying surgical instruments (Bistoury, Dissection Forceps, Straight Scissors, Curved Scissors) in cluttered tray environments.

Project Structure

create_yolo_dataset.py: Formats the original JSON annotations into individual YOLO-compatible .txt files.


verify_txt.py: Projects normalized YOLO coordinates back onto the images for visual verification.


eda_analysis.py: Generates class distribution and bounding box geometry charts.


train_yolo.py: The main training script, featuring customized hyperparameter tuning and data augmentation (Mosaic, Rotational, HSV).

generate_submission.py: Runs inference on the test dataset using best.pt and compiles the results into the required JSON format.

Requirements
To install the necessary dependencies, run:

Bash
pip install -r requirements.txt

(Ensure you have PyTorch installed with CUDA support if running on a GPU).

How to Run the Code
1. Data Preparation
Ensure the yolo_surgical_dataset folder is present in the root directory and contains the proper train and val splits.

2. Training the Model
To begin the 50-epoch training process, execute:

Bash
python train_yolo.py
This will generate a runs/detect/CE888_Runs/ directory containing the loss curves, confusion matrices, and the best model weights (best.pt).

3. Generating Test Predictions
To generate the final JSON submission file for the test dataset, run:

Bash
python generate_submission.py
This script will load the best.pt weights, run inference on the test images, and output a file formatted exactly to the specifications of sample_submission_test.json