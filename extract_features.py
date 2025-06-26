import torchreid
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

# Load model
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True,
    use_gpu=torch.cuda.is_available()
)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Folder with crops
crops_dir = 'runs/detect/predict2/crops'
features = []
filenames = []

# Process each image
for fname in os.listdir(crops_dir):
    if fname.endswith(".jpg"):
        img_path = os.path.join(crops_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            feat = model(img_tensor)
        features.append(feat.squeeze().numpy())
        filenames.append(fname)

# Save features
np.save('player_features.npy', np.array(features))
with open('filenames.txt', 'w') as f:
    for name in filenames:
        f.write(name + '\n')

print("✅ Features extracted and saved.")
