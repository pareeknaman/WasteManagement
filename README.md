# ♻️ Smart Waste Classification System

An AI-powered waste classification web application that uses **MobileNetV2** deep learning to identify waste types and provide smart disposal recommendations.

### 🌐 [Live Demo → wastemanagement-zadkzkkgepjacdpetjwdyo.streamlit.app](https://wastemanagement-zadkzkkgepjacdpetjwdyo.streamlit.app/)

---

## 📸 Features

- **12-Class Waste Classification** — Accurately classifies waste into: Battery, Biological, Brown Glass, Cardboard, Clothes, Green Glass, Metal, Paper, Plastic, Shoes, Trash, White Glass
- **Real-Time Camera Input** — Take a photo directly from your device camera for instant classification
- **Image Upload** — Upload JPG/JPEG/PNG images for classification
- **CV2 Visual Tracking** — Annotates the processed image with a green tracking circle using OpenCV
- **AI Disposal Guide** — Provides concise, expert disposal instructions based on the classified waste type
- **Confidence Scoring** — Visual confidence bar with high/moderate/low indicators
- **Full Probability Breakdown** — Expandable chart showing probabilities across all 12 classes

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) with custom CSS (glassmorphism, dark gradient theme) |
| **ML Model** | MobileNetV2 (TensorFlow/Keras) — transfer learning, 12-class output |
| **Computer Vision** | OpenCV (`opencv-python-headless`) for image annotation |
| **AI Integration** | LLM-powered disposal guide (configured via API key) |
| **Deployment** | Streamlit Community Cloud |

---

## 📂 Project Structure

```
PBL_ML_Project/
├── app.py                                  # Main Streamlit application
├── MobileNetV2 Waste Management.keras      # Trained classification model (~10 MB)
├── patch_model.py                          # Utility to strip quantization_config from model
├── requirements.txt                        # Python dependencies
├── .env                                    # Environment variables (Groq API key — not in repo)
├── .gitignore                              # Git ignore rules
├── .python-version                         # Python 3.11 pinned for deployment
└── README.md                               # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/pareeknaman/WasteManagement.git
cd WasteManagement

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Image Input │────▶│  Preprocessing   │────▶│   MobileNetV2   │
│ (Camera/Upload)    │  (Resize 224x224,│     │  Classification │
└──────────────┘     │   Normalize)     │     │  (12 classes)   │
                     └──────────────────┘     └────────┬────────┘
                                                       │
                     ┌──────────────────┐              │
                     │  OpenCV Tracking │◀─────────────┤
                     │  (Green Circle)  │              │
                     └──────────────────┘              ▼
                                              ┌─────────────────┐
                                              │   AI Disposal   │
                                              │     Guide       │
                                              └─────────────────┘
```

1. **Input** — User uploads an image or takes a photo via the camera
2. **Preprocessing** — Image is resized to 224×224 and normalized to [0, 1]
3. **Classification** — MobileNetV2 predicts probabilities across 12 waste categories
4. **Annotation** — OpenCV draws a green tracking circle on the original image
5. **AI Analysis** — The predicted class is sent to the AI model for expert disposal advice

---

## 📊 Supported Waste Categories

| # | Category | Emoji | Typical Items |
|---|----------|-------|---------------|
| 1 | Battery | 🔋 | AA, AAA, lithium-ion, button cells |
| 2 | Biological | 🌿 | Food scraps, yard waste, compostables |
| 3 | Brown Glass | 🟤 | Beer bottles, medicine bottles |
| 4 | Cardboard | 📦 | Shipping boxes, cereal boxes |
| 5 | Clothes | 👕 | T-shirts, jeans, fabric scraps |
| 6 | Green Glass | 🟢 | Wine bottles, juice bottles |
| 7 | Metal | ⚙️ | Cans, foil, scrap metal |
| 8 | Paper | 📄 | Newspapers, office paper, receipts |
| 9 | Plastic | 🧴 | Bottles, containers, bags |
| 10 | Shoes | 👟 | Sneakers, sandals, boots |
| 11 | Trash | 🗑️ | Non-recyclable mixed waste |
| 12 | White Glass | ⬜ | Clear glass jars, drinking glasses |

---

## 📦 Dependencies

```
streamlit
tensorflow-cpu
pillow
numpy>=1.23.5,<2.0.0
opencv-python-headless==4.8.1.78
pandas
groq
python-dotenv
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
