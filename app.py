import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Prevent TF threading issues on macOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    min-height: 100vh;
  }

  /* Card-style containers */
  .result-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
  }

  .result-card h3 { color: #a8e6cf; margin: 0 0 0.5rem 0; font-size: 1rem; letter-spacing: 1px; text-transform: uppercase; }
  .result-card .value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
  .result-card .sub  { font-size: 0.9rem; color: rgba(255,255,255,0.55); margin-top: 0.2rem; }

  /* Confidence bar */
  .conf-bar-wrap { margin-top: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 999px; height: 10px; overflow: hidden; }
  .conf-bar      { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #56ab2f, #a8e063); transition: width 0.6s ease; }

  /* Badge colours per confidence level */
  .badge-high   { color: #a8e6cf; }
  .badge-medium { color: #ffe082; }
  .badge-low    { color: #ef9a9a; }

  h1 { color: #ffffff !important; }
  .stSpinner > div { border-top-color: #56ab2f !important; }
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.markdown("# ♻️ Smart Waste Classification")
st.markdown("<p style='color:rgba(255,255,255,0.55); margin-top:-0.5rem;'>Powered by MobileNetV2 · 12-Class Model</p>", unsafe_allow_html=True)
st.divider()

# ─── Sidebar: API Key ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🤖 Groq AI Integration")
env_api_key = os.getenv("GROQ_API_KEY", "")
if env_api_key:
    groq_api_key = env_api_key
    st.sidebar.success("✅ API Key securely loaded from .env")
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    st.sidebar.caption("This unlocks the Smart Disposal Guide powered by Groq.")

# ─── AI Helper Function ──────────────────────────────────────────────────────
def get_disposal_advice(predicted_class, api_key, base64_image):
    client = Groq(api_key=api_key)
    prompt = f"You are a waste expert. The AI classified this mostly as {predicted_class}. Look at the image and reply in EXACTLY 2 sentences. The VERY FIRST line must be ONLY the category in bold (e.g., **Recyclable**). The second line must concisely state the exact disposal method. Do NOT provide scenarios or bullet points."
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]
    )
    return response.choices[0].message.content

# ─── Class Names (12 alphabetically sorted categories) ───────────────────────
class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# ─── Emoji map for friendlier display ────────────────────────────────────────
CLASS_EMOJI = {
    'battery': '🔋', 'biological': '🌿', 'brown-glass': '🟤',
    'cardboard': '📦', 'clothes': '👕', 'green-glass': '🟢',
    'metal': '⚙️', 'paper': '📄', 'plastic': '🧴',
    'shoes': '👟', 'trash': '🗑️', 'white-glass': '⬜',
}

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Model file has been patched (quantization_config stripped).
    # Load with compile=False to skip optimizer/loss config (inference only).
    import tensorflow as tf
    return tf.keras.models.load_model(
        "MobileNetV2 Waste Management.keras",
        compile=False
    )

with st.spinner("Loading AI Model…"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ─── Input Methods ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    camera_image = st.camera_input("📸 Take a Photo")
with col2:
    uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

# Prefer camera over upload if both provided
raw_input = camera_image if camera_image is not None else uploaded_file

# ─── Inference & Annotation ──────────────────────────────────────────────────
if raw_input is not None:
    # Open with PIL
    pil_image = Image.open(raw_input)

    # ── 1. Preprocess for MobileNetV2 ──
    with st.spinner("Classifying…"):
        rgb_pil = pil_image.convert("RGB")
        resized  = rgb_pil.resize((224, 224))
        img_arr  = np.array(resized, dtype=np.float32) / 255.0
        img_arr  = np.expand_dims(img_arr, axis=0)

        predictions  = model.predict(img_arr, verbose=0)[0]
        class_idx    = int(np.argmax(predictions))
        confidence   = float(predictions[class_idx])
        class_label  = class_names[class_idx]
        confidence_pct = confidence * 100

    # ── 2. Draw a CV2 tracking circle on the original image ──
    # Convert PIL → NumPy BGR for OpenCV
    orig_np  = np.array(rgb_pil)             # H x W x 3  (RGB)
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    h, w = orig_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 3

    # Bright green circle (BGR: 0, 255, 0), thickness 3
    annotated_bgr = cv2.circle(orig_bgr.copy(), (cx, cy), radius, (0, 255, 0), 3)

    # Convert back to RGB for st.image()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ── 3. Display annotated image ──
    st.image(annotated_rgb, caption="📍 CV2 Tracking Circle Applied", use_container_width=True)

    # ── 4. Result card ──
    emoji = CLASS_EMOJI.get(class_label, '♻️')

    if confidence_pct >= 75.0:
        badge_cls = "badge-high"
        verdict   = "✅ High Confidence"
    elif confidence_pct >= 50.0:
        badge_cls = "badge-medium"
        verdict   = "⚠️ Moderate Confidence"
    else:
        badge_cls = "badge-low"
        verdict   = "❓ Low Confidence — Try a clearer image"

    st.markdown(f"""
    <div class="result-card">
        <h3>Predicted Class</h3>
        <div class="value">{emoji} {class_label.title()}</div>
        <div class="sub {badge_cls}">{verdict}</div>
        <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:{confidence_pct:.1f}%"></div>
        </div>
        <div class="sub" style="margin-top:0.4rem;">Confidence: <strong>{confidence_pct:.2f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # ── 5. AI Disposal Guide ──
    st.markdown("### 💡 Smart Disposal Guide")
    if groq_api_key:
        with st.spinner('Analyzing...'):
            try:
                import base64
                from io import BytesIO
                buffered = BytesIO()
                rgb_pil.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                advice = get_disposal_advice(class_label, groq_api_key, img_str)
                st.info(advice)
            except Exception as e:
                st.error(f"Error fetching disposal advice: {e}")
    else:
        st.warning("Please enter your Groq API Key in the sidebar to unlock the Smart Disposal Guide.")

    # All 12 class probabilities as a bar chart
    with st.expander("📊 Full Probability Breakdown"):
        import pandas as pd
        probs_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": (predictions * 100).round(2)
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        st.dataframe(probs_df, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align:center; padding: 3rem; color: rgba(255,255,255,0.4);'>
        <p style='font-size:3rem;'>♻️</p>
        <p>Use the camera or upload an image to classify waste.</p>
    </div>
    """, unsafe_allow_html=True)
