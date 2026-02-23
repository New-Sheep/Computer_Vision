from ultralytics import YOLO
import os


def train_surgical_model():
    print("Initializing YOLOv8 Model Training...")

    # 1. Load a pre-trained baseline model.
    # 'yolov8s.pt' (Small) is a great balance between speed and accuracy.
    # It will automatically download the ~20MB weights file the first time you run this.
    model = YOLO('yolov8s.pt')

    # 2. Define the path to your data.yaml
    # Make sure this matches where your file actually is!
    yaml_path = os.path.join('yolo_surgical_dataset', 'data.yaml')

    # 3. Start Training
    results = model.train(
        data=yaml_path,
        epochs=50,  # Number of times the model will see the entire dataset
        imgsz=640,  # Resize images to 640x640 for training (standard practice)
        batch=16,  # Number of images processed at once (reduce to 8 if you get an OutOfMemory error)
        project='CE888_Runs',  # Folder where all your training results will be saved
        name='yolov8s_run1',  # Name of this specific training experiment

        # --- Advanced Data Augmentation Strategies ---
        # We add these to specifically handle the "cluttered tray environments"
        degrees=45.0,  # Rotate images up to 45 degrees (combats the rotational variance we saw in EDA)
        mosaic=1.0,  # Stitches 4 images together to teach the model to handle heavy clutter
        hsv_s=0.2,  # Slightly vary saturation to handle lighting changes on the metal trays
        hsv_v=0.2  # Slightly vary brightness to handle metallic reflections
    )

    print("\nTraining Complete! Check the 'CE888_Runs/yolov8s_run1' folder for your results and charts.")


if __name__ == '__main__':
    # We use this if __name__ block because multiprocessing (used by YOLO for data loading)
    # requires it on Windows machines to prevent crashing.
    train_surgical_model()