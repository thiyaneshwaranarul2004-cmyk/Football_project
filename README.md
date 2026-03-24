# ⚽ AI Football Match Analysis System

## 📌 Project Overview

This project is an AI-based football match analysis system that detects players and the ball from a video and provides real-time insights and strategy suggestions. It simulates a TV broadcast-style interface with live analytics.

---

## 🚀 Features

* 🎯 Player detection using YOLOv8
* ⚽ Ball tracking
* 🟥🟦 Team classification (Red vs Blue)
* 📊 Possession analysis (percentage)
* 👤 Unique player tracking IDs (e.g., Red_P1, Blue_P2)
* 🧠 AI-based strategy suggestions
* 📺 TV broadcast-style UI

---

## 🧠 AI Strategy Output

The system analyzes player movement and team possession to generate insights like:

* Which team should attack or defend
* Best performing player (high movement)
* Weak player (low activity)
* Tactical suggestions for winning the match

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Ultralytics YOLOv8
* Computer Vision & AI

---

## 📂 Project Structure

```
Football_Project/
│── football.py                # Main analysis code
│── footballclip.mp4           # Input video
│── yolov8n.pt                # YOLO model (download separately)
│── README.md                 # Project documentation
```

---

## ⚙️ Installation

1. Install Python (recommended: Python 3.10 or 3.11)

2. Install required libraries:

```
pip install opencv-python numpy ultralytics
```

3. Download YOLOv8 model:

* It will auto-download when running the code

---

## ▶️ How to Run

1. Open VS Code
2. Open the project folder
3. Open terminal
4. Run:

```
python football.py
```

---

## 📺 Output

* Video window will open
* Shows:

  * Player detection
  * Ball tracking
  * Team possession %
  * AI strategy suggestions

Press **ESC** to exit.

---

## ⚠️ Limitations

* Does not detect real player names or jersey numbers
* Works best with clear match videos
* Basic strategy logic (not professional-level analytics yet)

---

## 🔮 Future Improvements

* Player name recognition
* Pass detection system
* Formation analysis
* Heatmaps
* Advanced AI tactics

---

## 👨‍💻 Author

Your Name

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
