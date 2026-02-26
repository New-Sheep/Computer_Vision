from ultralytics import YOLO
import json
import os
import glob

def generate_test_predictions():
    print("Loading the best trained model...")
    # UPDATE THIS PATH to match your final run folder!
    # Based on your logs, it looks like it saved to yolov8s_run12
    model_path = os.path.join('runs', 'detect', 'CE888_Runs', 'yolov8s_run12', 'weights', 'best.pt')
    model = YOLO(model_path)

    # Path to your 2107 test images
    test_images_dir = os.path.join('CE888-data-resit', 'Images', 'Test', 'images')
    image_paths = glob.glob(os.path.join(test_images_dir, '*.jpg'))
    
    print(f"Found {len(image_paths)} test images. Starting inference...")
    
    submission_dict = {}

    # Run predictions on all images
    for img_path in image_paths:
        # Get the filename and replace .jpg with .txt as required by the sample format
        filename = os.path.basename(img_path)
        txt_key = filename.replace('.jpg', '.txt')
        
        # Run YOLO inference
        results = model(img_path, verbose=False)[0]
        
        labels_list = []
        
        # Iterate over all detected boxes in this image
        for box in results.boxes:
            # YOLO outputs xywhn (normalized x_center, y_center, width, height)
            x_c, y_c, w, h = box.xywhn[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Format exactly as required
            detection = {
                "class": cls_id,
                "x_center": round(x_c, 6),
                "y_center": round(y_c, 6),
                "width": round(w, 6),
                "height": round(h, 6),
                "confidence": round(conf, 4)
            }
            labels_list.append(detection)
            
        submission_dict[txt_key] = {"labels": labels_list}

    # Save to JSON
    output_filename = "final_submission_test.json"
    with open(output_filename, 'w') as f:
        json.dump(submission_dict, f, indent=2)
        
    print(f"\nSuccess! Generated {output_filename} containing predictions for {len(submission_dict)} images.")

if __name__ == '__main__':
    generate_test_predictions()