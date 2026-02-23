import cv2
import json
import matplotlib.pyplot as plt
import os


def visualize_yolo_json_improved(json_path, image_dir, image_key, class_map):
    # Load the JSON labels
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if the key exists in the JSON
    if image_key not in data:
        print(f"Error: '{image_key}' not found in {json_path}")
        return

    # Get the image filename (replacing .txt with .jpg)
    image_filename = image_key.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_filename)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Define distinct colors for each class (in RGB format)
    colors = {
        0: (255, 50, 50),  # Red (Bistoury)
        1: (50, 255, 50),  # Green (Dissection Forceps)
        2: (50, 100, 255),  # Blue (Straight Scissors)
        3: (255, 200, 50),  # Yellow (Curved Scissor)
        4: (255, 50, 255)  # Magenta (For the anomalous Class 4)
    }

    # Draw boxes
    for obj in data[image_key]['labels']:
        cls_id = obj['class']
        label = class_map.get(cls_id, f"Class {cls_id}")
        color = colors.get(cls_id, (255, 255, 255))  # Default to white if unknown class

        # Denormalize coordinates
        x_c, y_c = obj['x_center'] * w, obj['y_center'] * h
        bw, bh = obj['width'] * w, obj['height'] * h

        # Calculate top-left and bottom-right corners
        x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
        x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)

        # 1. Draw a thinner bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # 2. Add a solid background rectangle for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Get the width and height of the text box
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size

        # Prevent the background box from going off the top of the image
        bg_y1 = max(0, y1 - text_h - 10)
        bg_y2 = y1 if y1 > (text_h + 10) else (text_h + 10)

        # Draw the filled background rectangle (-1 thickness means filled)
        cv2.rectangle(img, (x1, bg_y1), (x1 + text_w + 4, bg_y2), color, -1)

        # 3. Add the text over the background
        # Use black text for yellow backgrounds to keep contrast high, white for others
        text_color = (0, 0, 0) if cls_id == 3 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 2, bg_y2 - 4), font, font_scale, text_color, font_thickness)

    # Plotting the image
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.title(f"Improved Visualization: {image_filename}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- Usage ---
CLASS_NAMES = {
    0: "Bistoury",
    1: "Dissection Forceps",
    2: "Straight Scissors",
    3: "Curved Scissor"
}

# Run it on your specific file!
# (Make sure to update the 'image_dir' parameter to where your images are saved)
visualize_yolo_json_improved(
    json_path='CE888-data-resit/Labels/labels_train.json',
    image_dir='CE888-data-resit/Images/Train/images',
    image_key='85pir8mfug.txt',
    class_map=CLASS_NAMES
)