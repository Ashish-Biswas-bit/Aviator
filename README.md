# 🛩️ Aviator Score Analyzer & Predictor

A desktop application that predicts the next multiplier score in the Aviator game using **XGBoost machine learning**. Supports both **manual input** and **OCR-based extraction** from screenshots via **Tesseract OCR**.

---

## ✨ Features

- **🔢 Manual Input** — Paste scores directly as text (comma, space, or newline separated)
- **📷 OCR Extraction** — Paste from clipboard or browse an image, extract scores automatically using Tesseract OCR
- **🤖 XGBoost Prediction** — Sliding-window feature engineering trains a model on your history to predict the next multiplier
- **💾 Persistent Database** — All scores saved locally in SQLite — your history persists across sessions
- **🌗 Dark / Light Mode** — Automatically follows your system theme via CustomTkinter
- **📜 Scrollable Tabs** — Fully responsive window; content scrolls if the window is too small

---

## 🧰 Tech Stack

| Component            | Technology                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| **GUI**              | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) (Python)                                        |
| **OCR**              | [pytesseract](https://github.com/madmaze/pytesseract) + [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| **Image Processing** | [OpenCV](https://opencv.org/) + [Pillow](https://python-pillow.org/)                                            |
| **ML Model**         | [XGBoost](https://xgboost.readthedocs.io/) regressor                                                            |
| **Database**         | [SQLite3](https://docs.python.org/3/library/sqlite3.html)                                                       |

---

## 🚀 Installation

### 1. Install Tesseract OCR

Download and install **Tesseract OCR** from GitHub:

👉 [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

> **Default path**: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 2. Setup Python environment

```bash
# Clone the repository
git clone https://github.com/ARAFATxPRO/aviator-predictor.git
cd aviator-predictor

# (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

---

## 📖 Usage

### Manual Input Tab

1. Type or paste scores into the text area (e.g. `1.23, 5.40, 2.10`)
2. Click **🚀 Process & Predict**
3. The predicted next score and model training status are displayed below

### Image / Screenshot Tab

1. Click **📋 Paste from Clipboard** or **📁 Browse Image File** to load a screenshot
2. Click **🔍 Extract Scores & Predict**
3. Scores are automatically parsed via OCR and fed into the prediction model

### Database Management

- Click **🗑️ Clear Database** (in either tab) to wipe all stored history
- Recent scores are displayed in the bottom panel with total count

---

## 📂 Project Structure

```
aviator-predictor/
├── main.py              # Entry point
├── gui.py               # CustomTkinter GUI (scrollable tabs, modern design)
├── ocr_utils.py         # Tesseract OCR pipeline (preprocessing + extraction)
├── ml_model.py          # XGBoost model (sliding window features)
├── database.py          # SQLite CRUD operations
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

---

## ⚙️ How It Works

1. **Input** — Scores are obtained via manual text entry or OCR from a screenshot
2. **Storage** — Scores are saved to a local SQLite database (`score_analyzer.db`)
3. **Feature Engineering** — The model uses a sliding window of the last 3 scores plus a rolling mean
4. **Training** — XGBoost regressor is trained on all available history
5. **Prediction** — The next multiplier score is predicted and displayed

---

## 🧪 Requirements

- Python 3.8+
- Tesseract OCR (v5.0+ recommended)
- Windows / macOS / Linux

See [`requirements.txt`](requirements.txt) for full Python dependency list.

---

## 📄 License

This project is for educational and personal use only.
