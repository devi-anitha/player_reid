# Player Re-Identification in Sports Footage 🎥⚽

## Company: Liat.ai  
**Role:** AI Intern  
**Assignment:** Re-Identification in a Single Feed  
**Submitted by:** Anitha Reddy

---

## 🧠 Overview

This project implements a solution for player **re-identification in a single video feed**, where the objective is to assign consistent IDs to players who exit and re-enter the frame during gameplay. This simulates real-time player tracking and re-identification under partial occlusion or motion gaps.

The model ensures that the **same player retains the same ID**, even after disappearing and reappearing later in the video.

---

## 📁 Project Structure

```bash
player_reid/
├── extract_players.py            # Extracts cropped player images from detected frames
├── extract_features.py           # Extracts appearance features using Torchreid
├── reid_matching.py              # Matches and groups players based on similarity
├── requirements.txt              # All dependencies
├── README.md                     # This documentation
├── report.pdf                    # Report outlining approach & findings
├── runs/
│   └── detect/
│       └── predict2/
│           ├── frames/           # Extracted video frames
│           ├── crops/           # Player crop images
│           ├── features.npy     # Saved player features
│           └── reid_groups/     # Grouped players with consistent IDs
│── 15sec_input_720p.mp4      # Input video (single feed)
│──yolo8n.pt
