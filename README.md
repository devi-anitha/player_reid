# Player Re-Identification in Sports Footage 🎥⚽

## Company: Liat.ai  
**Role:** AI Intern  
**Assignment:** Re-Identification in a Single Feed  
**Submitted by:** Anitha Reddy

---

## 🧠 Overview

This project implements a solution for player **re-identification in a single video feed**, where the objective is to assign consistent IDs to players who exit and re-enter the frame during gameplay. This simulates real-time player tracking and re-identification under partial occlusion or motion gaps.

The pipeline detects players, extracts visual features, and groups the same player across frames using appearance-based similarity.

---

## 🔧 Technologies Used

- **YOLOv8** (Ultralytics) – for player detection
- **Torchreid** – for feature extraction and deep re-ID modeling
- **NumPy + scikit-learn** – for feature matching and cosine similarity
- **Python 3.10+**

---

## 📁 Project Structure

```bash
player_reid/
├── extract_players.py            # Extracts cropped player images from detected frames
├── extract_features.py           # Extracts appearance features using Torchreid
├── reid_matching.py              # Matches and groups players based on similarity
├── requirements.txt              # All Python dependencies
├── README.md                     # Project documentation
├── report.pdf                    # Full project report
├── yolo8n.pt                     # Pre-trained YOLOv8 model weights
├── 15sec_input_720p.mp4          # Input video used in the pipeline
├── runs/
│   └── detect/
│       └── predict2/
│           ├── frames/           # Extracted video frames
│           ├── crops/            # Cropped player images
│           ├── features.npy      # Saved appearance features
│           └── reid_groups/      # Final grouped results (same-ID clusters)
