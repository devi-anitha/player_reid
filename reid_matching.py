import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Load extracted features
features = np.load('player_features.npy')  # or .pt if saved with torch
image_names = sorted(os.listdir('runs/detect/predict2/crops'))

# Compute similarity matrix
similarity_matrix = cosine_similarity(features)

# Set threshold to consider same person
threshold = 0.9

# Group similar players
groups = []
visited = set()

for i in range(len(features)):
    if i in visited:
        continue
    group = [image_names[i]]
    visited.add(i)
    for j in range(i + 1, len(features)):
        if j not in visited and similarity_matrix[i][j] > threshold:
            group.append(image_names[j])
            visited.add(j)
    groups.append(group)

# Save results
output_path = 'reid_groups'
os.makedirs(output_path, exist_ok=True)

for idx, group in enumerate(groups):
    group_dir = os.path.join(output_path, f'player_{idx}')
    os.makedirs(group_dir, exist_ok=True)
    for image_name in group:
        src = os.path.join('runs/detect/predict2/crops', image_name)
        dst = os.path.join(group_dir, image_name)
        cv2.imwrite(dst, cv2.imread(src))

print(f"✅ Re-ID complete. Grouped images saved in: {output_path}")
