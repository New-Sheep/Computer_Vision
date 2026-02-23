import cv2
import matplotlib.pyplot as plt
import os


def verify_yolo_txt(image_path, txt_path, class_map):
    # 1. Load the image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 2. Load the text file
    if not os.path.exists(txt_path):
        print(f"Error: Label file not found at {txt_path}")
        return

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} objects in {txt_path}")

    # Define colors
    colors = {0: (255, 50, 50), 1: (50, 255, 50), 2: (50, 100, 255), 3: (255, 200, 50)}

    # 3. Parse each line and draw
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

        label = class_map.get(cls_id, f"Class {cls_id}")
        color = colors.get(cls_id, (255, 255, 255))

        # Denormalize coordinates to pixel values
        x_c, y_c = x_center * w, y_center * h
        bw, bh = bbox_width * w, bbox_height * h

        # Calculate corners
        x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
        x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)

        # Draw box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 4. Show the result
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(f"Verification from .txt: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- Run the Verification ---
CLASS_NAMES = {0: "Bistoury", 1: "Dissection Forceps", 2: "Straight Scissors", 3: "Curved Scissor"}

# Update these paths to match your directory structure exactly
test_image = 'CE888-data-resit/Images/Train/images/85pir8mfug.jpg'
test_txt = 'CE888-data-resit/Images/Train/labels/85pir8mfug.txt'

verify_yolo_txt(test_image, test_txt, CLASS_NAMES)