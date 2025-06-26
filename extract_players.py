import os
import cv2

# Path to YOLO detection labels (txt files)
labels_path = "runs/detect/predict2/labels"

# Path to original frames (00000.jpg, 00001.jpg, etc.)
frames_path = "runs/detect/predict2/frames"

# Output folder for cropped players
output_path = "runs/detect/predict2/crops"
os.makedirs(output_path, exist_ok=True)

# Iterate over all label files
for label_file in os.listdir(labels_path):
    if label_file.endswith(".txt"):
        # Extract frame number from label filename (e.g., 15sec_input_720p_65.txt → 65)
        frame_index = int(label_file.split("_")[-1].split(".")[0])
        image_name = f"{frame_index:05d}.jpg"  # format as 00065.jpg
        image_path = os.path.join(frames_path, image_name)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠️  Frame not found: {image_path}")
            continue

        # Load image
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Read label file
        with open(os.path.join(labels_path, label_file), "r") as f:
            lines = f.readlines()

        # Process each detection
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed line
            cls, x_center, y_center, w, h = map(float, parts)

            # Convert YOLO coords to pixel coords
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            # Clip bounding box to image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Crop and save
            crop = image[y1:y2, x1:x2]
            crop_filename = f"{os.path.splitext(image_name)[0]}_crop{i}.jpg"
            cv2.imwrite(os.path.join(output_path, crop_filename), crop)

print("✅ Done. Crops saved.")
