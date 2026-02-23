import json
import matplotlib.pyplot as plt
import pandas as pd


def run_eda(json_path, class_map):
    print("Loading data for EDA...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Lists to hold our extracted data
    class_counts = {cls: 0 for cls in class_map.values()}
    widths = []
    heights = []

    # Parse the JSON
    for img_key, content in data.items():
        for obj in content.get('labels', []):
            cls_id = obj['class']
            # Map the ID to the name, ignoring anomalous classes like '4' for now
            if cls_id in class_map:
                cls_name = class_map[cls_id]
                class_counts[cls_name] += 1
                widths.append(obj['width'])
                heights.append(obj['height'])

    # --- Plot 1: Class Distribution ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.title('Class Distribution in Training Set', fontsize=16)
    plt.xlabel('Surgical Instrument Class', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add exact numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 10, int(yval), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)  # Saves a high-res copy for your report
    plt.show()

    # --- Plot 2: Bounding Box Size Distribution ---
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.3, color='purple', edgecolors='none')
    plt.title('Bounding Box Size Distribution (Normalized)', fontsize=16)
    plt.xlabel('Normalized Width', fontsize=12)
    plt.ylabel('Normalized Height', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('bbox_distribution.png', dpi=300)  # Saves a high-res copy for your report
    plt.show()

    print("EDA complete! Charts have been saved as high-resolution PNGs.")


# --- Execute EDA ---
CLASS_NAMES = {
    0: "Bistoury",
    1: "Dissection Forceps",
    2: "Straight Scissors",
    3: "Curved Scissor"
}

train_json_path = 'CE888-data-resit/Labels/labels_train.json'
run_eda(train_json_path, CLASS_NAMES)