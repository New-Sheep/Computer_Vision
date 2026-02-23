import os
import random
import shutil


def setup_yolo_dataset(src_img_dir, src_lbl_dir, base_out_dir, split_ratio=0.8, seed=42):
    """
    Randomly splits images and labels into train/val sets and generates data.yaml.
    """
    print(f"Setting up YOLO dataset in '{base_out_dir}'...")

    # 1. Define output directories
    dirs = {
        'train_img': os.path.join(base_out_dir, 'train', 'images'),
        'train_lbl': os.path.join(base_out_dir, 'train', 'labels'),
        'val_img': os.path.join(base_out_dir, 'val', 'images'),
        'val_lbl': os.path.join(base_out_dir, 'val', 'labels')
    }

    # Create the directories safely
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # 2. Get all images and shuffle them
    all_images = [f for f in os.listdir(src_img_dir) if f.endswith('.jpg')]

    # Set a random seed so you get the exact same split if you run this again
    random.seed(seed)
    random.shuffle(all_images)

    # 3. Calculate split index
    split_idx = int(len(all_images) * split_ratio)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    print(f"Total images found: {len(all_images)}")
    print(f"Allocating {len(train_imgs)} for Training (80%)")
    print(f"Allocating {len(val_imgs)} for Validation (20%)")

    # 4. Helper function to copy files
    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        missing_labels = 0
        for img_name in file_list:
            # Copy Image
            src_img = os.path.join(src_img_dir, img_name)
            dst_img = os.path.join(dest_img_dir, img_name)
            shutil.copy2(src_img, dst_img)

            # Copy corresponding Label
            lbl_name = img_name.replace('.jpg', '.txt')
            src_lbl = os.path.join(src_lbl_dir, lbl_name)
            dst_lbl = os.path.join(dest_lbl_dir, lbl_name)

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                missing_labels += 1
                # If an image has no objects, YOLO expects an empty txt file
                open(dst_lbl, 'a').close()

        return missing_labels

    # 5. Execute the copy
    print("\nCopying Training files...")
    train_missing = copy_files(train_imgs, dirs['train_img'], dirs['train_lbl'])

    print("Copying Validation files...")
    val_missing = copy_files(val_imgs, dirs['val_img'], dirs['val_lbl'])

    if train_missing > 0 or val_missing > 0:
        print(f"Note: Created empty label files for {train_missing + val_missing} background images.")

    # 6. Generate the data.yaml file
    yaml_path = os.path.join(base_out_dir, 'data.yaml')
    yaml_content = f"""# YOLOv8 Dataset Configuration File
path: {os.path.abspath(base_out_dir)} # Absolute path to dataset root
train: train/images # relative to 'path'
val: val/images # relative to 'path'

# Classes
names:
  0: Bistoury
  1: Dissection Forceps
  2: Straight Scissors
  3: Curved Scissor
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nSuccess! Setup complete. Your data.yaml is located at: {yaml_path}")


# --- Execute Setup ---
# Your existing source paths
SOURCE_IMAGES = 'CE888-data-resit/Images/Train/images'
SOURCE_LABELS = 'CE888-data-resit/Images/Train/labels'

# The new, clean directory where we will assemble the final ML dataset
OUTPUT_DATASET_DIR = 'yolo_surgical_dataset'

setup_yolo_dataset(SOURCE_IMAGES, SOURCE_LABELS, OUTPUT_DATASET_DIR)