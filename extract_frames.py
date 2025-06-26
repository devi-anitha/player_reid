import cv2
import os

video_path = "15sec_input_720p.mp4"
output_folder = "runs/detect/predict2/frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(output_folder, f"{i:05}.jpg"), frame)
    i += 1

cap.release()
print(f"✅ Saved {i} frames to {output_folder}")
