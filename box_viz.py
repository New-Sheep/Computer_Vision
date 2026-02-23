import cv2
import json
import matplotlib.pyplot as plt
import os


def visualize_yolo_json(json_path, image_dir, image_key, class_map):
    # Load the JSON labels
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get the image filename (replacing .txt with .jpg if necessary)
    image_filename = image_key.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_filename)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Draw boxes
    for obj in data[image_key]['labels']:
        cls_id = obj['class']
        label = class_map.get(cls_id, f"Class {cls_id}")

        # Denormalize coordinates [cite: 16]
        # YOLO format: x_center, y_center, width, height
        x_c, y_c = obj['x_center'] * w, obj['y_center'] * h
        bw, bh = obj['width'] * w, obj['height'] * h

        # Calculate top-left corner
        x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
        x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)

        # Draw rectangle and text
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 0, 0), 4)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(f"Visualizing: {image_filename}")
    plt.axis('off')
    plt.show()


# --- Usage ---
CLASS_NAMES = {0: "Bistoury", 1: "Dissection Forceps", 2: "Straight Scissors", 3: "Curved Scissor"}

# Example for a training image
visualize_yolo_json(
    json_path='CE888-data-resit/Labels/labels_train.json',
    image_dir='CE888-data-resit/Images/Train/images',
    image_key='85pir8mfug.txt',
    class_map=CLASS_NAMES
)