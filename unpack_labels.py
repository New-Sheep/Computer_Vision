import json
import os


def convert_json_to_yolo(json_path, output_labels_dir):
    """
    Reads a JSON file containing bounding box data and creates
    individual YOLO format .txt files.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_labels_dir, exist_ok=True)

    # 2. Load the JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return

    print(f"Loaded {len(data)} entries from {json_path}")

    # 3. Iterate through every image key (e.g., '85pir8mfug.txt')
    count = 0
    for txt_filename, content in data.items():
        # Create the full path for the new .txt file
        txt_filepath = os.path.join(output_labels_dir, txt_filename)

        # Open the file and write each bounding box on a new line
        with open(txt_filepath, 'w') as out_f:
            for obj in content.get('labels', []):
                cls_id = obj['class']
                x_c = obj['x_center']
                y_c = obj['y_center']
                w = obj['width']
                h = obj['height']

                # YOLO format: class x_center y_center width height
                # Space-separated, 6 decimal places for precision
                out_f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        count += 1

    print(f"Successfully created {count} YOLO label files in '{output_labels_dir}'!\n")


# ==========================================
# Execute the Conversion
# ==========================================

# 1. Convert Training Labels
train_json_path = 'CE888-data-resit/Labels/labels_train.json'
train_labels_dir = 'CE888-data-resit/Images/Train/labels'

print("--- Converting Training Labels ---")
convert_json_to_yolo(train_json_path, train_labels_dir)

# 2. Convert Test Sample Labels
# Note: The test set usually doesn't have ground truth labels for you to train on,
# but unpacking the sample helps verify the format.
test_json_path = 'CE888-data-resit/Labels/labels_test_sample.json'
test_labels_dir = 'CE888-data-resit/Images/Test/labels'

print("--- Converting Test Sample Labels ---")
convert_json_to_yolo(test_json_path, test_labels_dir)