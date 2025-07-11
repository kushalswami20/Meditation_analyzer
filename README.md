````markdown
# 🧘 Meditation Analyzer

A deep learning-based system that analyzes a video input to determine whether a person is meditating. It combines insights from facial expressions, emotional state (vision + audio), body posture, and breathing rhythm using specialized neural network models.

---

## 🔍 Overview

This project leverages multiple deep learning models working in parallel to classify meditation status from video (and optionally audio) input. Each model is responsible for analyzing a different aspect of the human state:

- **Face Expression** — Calm vs Not-Calm
- **Emotion (Vision & Audio)** — Emotionally relaxed or stressed
- **Posture** — Whether the meditation posture is correct
- **Breathing** — Regular and deep breathing patterns

<img width="3940" height="2970" alt="ec7d26e2" src="https://github.com/user-attachments/assets/d2a9f018-3978-44dc-8a89-60934511da92" />
All outputs are combined using a **Decision Fusion System** to make a final meditation prediction.

---

## 🧠 Deep Learning Models Used

| Component           | Model Architecture                   | Task                                      |
|---------------------|--------------------------------------|-------------------------------------------|
| Face Expression     | MobileNet-V3-Small (CNN)             | Calm/Not-Calm facial classification       |
| Emotion (Vision)    | ResNet-18 + Squeeze-Excite           | Emotion detection from facial cues        |
| Emotion (Audio)     | 3-layer Bidirectional GRU            | Emotion detection from audio              |
| Posture             | 2-layer Bi-LSTM + Attention          | Posture classification from body keypoints|
| Breathing           | 1D CNN                               | Breathing regularity detection            |
| Fusion              | Weighted Average / Logistic Regression | Final meditation prediction              |

---

## 📈 Training Performance

Each model was trained independently and evaluated on both training and validation data. Below are visualizations of the training progress over 100 epochs.

### ✅ Accuracy Graphs

<img width="2400" height="1600" alt="72034469" src="https://github.com/user-attachments/assets/9fcc34de-9ab1-49ce-b966-96072b87cb92" />

- **Face Expression Accuracy**:  
  - Training: ~95%  
  - Validation: ~88%

- **Emotional Analyzer Accuracy**:  
  - Training: ~94%  
  - Validation: ~87%

---

### 📉 Loss Graphs

- **Posture Analyzer Loss**: Dropped from **1.6 to <0.1**
- **Breathing Analyzer Loss**: Training and validation loss dropped from **~1.8 to <0.1**

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/meditation-analyzer.git
   cd meditation-analyzer
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run model training (example for Face Expression model):

   ```bash
   python train_face_expression.py
   ```

4. Run full prediction on a video:

   ```bash
   python predict_meditation.py --video input.mp4
   ```

---

## 📂 Folder Structure

```
meditation-analyzer/
├── models/
│   ├── face_expression/
│   ├── emotion/
│   ├── posture/
│   └── breathing/
├── data/
├── utils/
├── screenshots/
│   ├── accuracy_graphs.png
│   └── loss_graphs.png
├── train_face_expression.py
├── predict_meditation.py
└── README.md
```

---

## ✨ Future Work

* Integrate real-time webcam inference.
* Add lightweight deployment via ONNX or TensorFlow Lite.
* Build a web dashboard to visualize inference results and feedback.

---

## 👤 Author

**Kushal Arogyaswami**
*B.Tech IT, Delhi Technological University*

---

## 📜 License

MIT License — use freely with credit.

```

---

Would you like me to generate placeholder images for the graphs or a badge-style project summary for the top of the README?
```
