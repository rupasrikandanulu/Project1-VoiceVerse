import os
import requests
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from collections import Counter
from PIL import Image  # ✅ Added for logo handling

# ---------------- Load API Key ----------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    st.error("HF_API_KEY not found. Put HF_API_KEY=your_key in .env and restart.")
    st.stop()

client = InferenceClient(token=HF_API_KEY)

# ---------------- Helper Functions ----------------
def hf_infer_http(prompt: str, model: str, max_tokens: int = 400, timeout: int = 90) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)

def rewrite_text(text: str, tone: str) -> str:
    prompt = f"Rewrite this text in a {tone} tone:\n\n{text}"
    models_to_try = [
        "ibm-granite/granite-3.2-8b-instruct",
        "ibm-granite/granite-3b-instruct"
    ]
    for model in models_to_try:
        try:
            return hf_infer_http(prompt, model).strip()
        except Exception:
            continue
    return text

def generate_audio(text, style):
    try:
        audio_bytes = client.text_to_speech(
            model="espnet/kan-bayashi_ljspeech_vits",  # ✅ supported TTS model
            text=f"{style} style {text}"
        )
        return audio_bytes
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None


def analyze_text(text: str):
    words = text.split()
    sentences = text.split(".")
    vocab = set(words)
    avg_len = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len(sentences))
    score = round((len(vocab)/len(words))*100 + avg_len, 2)
    return {"words": len(words), "unique_words": len(vocab), "avg_sentence_len": avg_len, "complexity": score}

# ---------------- Page Styling ----------------
st.set_page_config(page_title="EchoVerse 🎧", layout="centered")

st.markdown("""
    <style>
    body {background-color: #fffdfc;}
    .stButton button {
        background: #fbc2eb; 
        color: black; font-weight: 600; 
        border-radius: 10px; padding: 0.6em 1.2em;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    .stButton button:hover {opacity:0.95; transform: scale(1.01);}
    .stTextInput textarea, .stTextInput input {
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- App UI ----------------
col_logo, col_title = st.columns([1, 5])
with col_logo:
    # ✅ Safe logo loader
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=60)
    else:
        st.write("🔗")  # fallback placeholder
with col_title:
    st.markdown("<h1 style='color:#ff6ec7;'>EchoVerse 🎧</h1>", unsafe_allow_html=True)
st.write("<p style='color:gray;'>AI Audiobook Generator – Rewrite, Narrate, and Share ✨</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox("🎭 Choose tone", ["Neutral", "Suspenseful", "Inspiring", "Excited"])
with col2:
    style = st.selectbox("🤖 Narrator", ["Wise Mentor 🧙", "Energetic 🤩", "Calm 🌙", "Robotic 🤖"])

language = st.selectbox("🌍 Narration Language", ["English", "Hindi", "Telugu", "Spanish", "French"])

uploaded_file = st.file_uploader("📂 Upload .txt", type="txt")
text_input = st.text_area("Or paste your text ✨")

text = uploaded_file.read().decode("utf-8") if uploaded_file else text_input

if "rewritten_text" in st.session_state:
    style = "Narrative"  # or get style from user input
    audio_file = generate_audio(st.session_state.rewritten_text, style)

    if audio_file:
        st.audio(audio_file, format="audio/wav")

colA, colB = st.columns(2)
with colA:
    if st.button("✨ Rewrite Text"):
        if not text:
            st.warning("Please upload or paste text!")
        else:
            with st.spinner("✨ Rewriting..."):
                st.session_state.rewritten_text = rewrite_text(text, tone)
with colB:
    if st.button("🎧 Generate Audio"):
        if not st.session_state.rewritten_text:
            st.warning("Rewrite first ✨")
        else:
            with st.spinner("🎧 Generating audio..."):
                audio_file = generate_audio(st.session_state.rewritten_text, style)
                st.audio(audio_file)
                with open(audio_file, "rb") as f:
                    st.download_button("⬇️ Download MP3", f, "EchoVerse.mp3")

if st.session_state.rewritten_text:
    st.subheader("📝 Rewritten Text")
    st.write(st.session_state.rewritten_text)

    st.subheader("📊 Insights")
    stats = analyze_text(st.session_state.rewritten_text)
    st.json(stats)

    # Cute chart
    fig, ax = plt.subplots()
    ax.bar(stats.keys(), stats.values(), color=["#fbc2eb"] * len(stats))
    st.pyplot(fig)
