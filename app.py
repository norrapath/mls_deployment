import json
import subprocess
import sys
import uuid
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

SAMPLE_RATE = 32_000
DURATION = 3.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320
FMIN = 20
FMAX = SAMPLE_RATE // 2
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "5_rating.pt"
OUTPUT_DIR = Path("birdclef-2026/outputTrim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LSTMClassifier(nn.Module):
    def __init__(self, n_mels, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, num_classes),
        )

    def forward(self, x):
        sequence, _ = self.lstm(x)
        pooled = torch.cat([sequence.mean(dim=1), sequence.amax(dim=1)], dim=1)
        return self.classifier(pooled)


@st.cache_resource
def load_model_and_classes():
    """Loads the model and class labels once and caches them in memory."""
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    detected_num_classes = state_dict["classifier.2.weight"].shape[0]

    model = LSTMClassifier(
        n_mels=N_MELS,
        hidden_size=192,
        num_layers=2,
        num_classes=detected_num_classes,
        dropout=0.30,
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(checkpoint, dict) and "label_encoder_classes" in checkpoint:
        classes = checkpoint["label_encoder_classes"]
    else:
        classes = [f"Species_{i}" for i in range(detected_num_classes)]

    return model, classes


def trim_audio_file(input_path: Path) -> Path:
    """Trims audio input via trim.py using the active Python interpreter."""
    clean_stem = input_path.stem.replace("_trimmed", "")
    unique_suffix = uuid.uuid4().hex[:6]
    trimmed_path = OUTPUT_DIR / f"{clean_stem}_{unique_suffix}_trimmed.ogg"

    cmd = [sys.executable, "trim.py", str(input_path), str(trimmed_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"trim.py failed:\n{result.stderr}")

    return trimmed_path


def load_fixed_audio(path):
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if len(waveform) < TARGET_SAMPLES:
        waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))
    elif len(waveform) > TARGET_SAMPLES:
        max_start = len(waveform) - TARGET_SAMPLES
        start = max_start // 2
        waveform = waveform[start : start + TARGET_SAMPLES]

    return waveform


def waveform_to_logmel(waveform):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True)
    log_mel = (log_mel - mean) / (std + 1e-6)
    return torch.from_numpy(log_mel.T.copy())


model, classes = load_model_and_classes()

st.set_page_config(
    page_title="Bird Species Audio Classifier",
    page_icon="🐦",
    layout="wide",
)

st.sidebar.title("⚙️ Model Settings")
top_k = st.sidebar.slider("Top K Predictions", min_value=1, max_value=20, value=5, step=1)

st.title("🐦 Bird Species Audio Classifier")
st.markdown("Upload an audio recording or use your microphone to classify species with confidence scores.")
st.divider()

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Audio Input")
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "flac", "m4a"])
    audio_mic = st.audio_input("Or Record Live Audio")

    selected_audio = audio_mic if audio_mic is not None else audio_file

    if selected_audio:
        st.audio(selected_audio)

with col_output:
    st.subheader("📊 Classification Results")

    if selected_audio is not None:
        with st.spinner("Trimming audio and analyzing features..."):
            try:
                # Save input audio buffer
                temp_path = Path("temp_audio_input.wav")
                with open(temp_path, "wb") as f:
                    f.write(selected_audio.read())

                trimmed_audio_path = trim_audio_file(temp_path)

                waveform = load_fixed_audio(trimmed_audio_path)
                features = waveform_to_logmel(waveform).unsqueeze(0).to(DEVICE)

                with torch.inference_mode():
                    probabilities = torch.softmax(model(features), dim=1).squeeze(0).cpu().numpy()

                top_k_count = min(int(top_k), len(probabilities))
                indices = np.argsort(probabilities)[::-1][:top_k_count]

                ranked = [
                    {
                        "scientific_name": str(classes[i]),
                        "probability": float(probabilities[i]),
                        "rank": idx + 1,
                    }
                    for idx, i in enumerate(indices)
                ]

                top_pred = ranked[0]

                st.metric(
                    label="Top Identified Species",
                    value=top_pred["scientific_name"],
                    delta=f"{top_pred['probability'] * 100:.2f}% Confidence",
                )

                # Visual Output Tabs
                tab_chart, tab_json = st.tabs(["📈 Confidence Chart", "📄 Raw JSON"])

                with tab_chart:
                    chart_data = {item["scientific_name"]: item["probability"] for item in ranked}
                    st.bar_chart(chart_data, y_label="Probability", x_label="Species")

                with tab_json:
                    result_json = {
                        "scientific_name": top_pred["scientific_name"],
                        "probability": top_pred["probability"],
                        "top_predictions": ranked,
                    }
                    st.json(result_json)

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

    else:
        st.info("👈 Please upload or record an audio file to view predictions.")
