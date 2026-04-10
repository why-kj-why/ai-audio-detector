import streamlit as st
import torch
import torchaudio
import io
from transformers import Wav2Vec2Processor
from train_v1 import CNN
from train_v2 import Model


DEVICE = "cuda" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 4
N_MELS = 64

st.set_page_config(page_title="AI Voice Detector", layout="centered")

st.title("🎙️ AI Voice Detector")
st.write("Detect whether an audio clip is REAL or AI GENERATED")


@st.cache_resource
def load_models():
    cnn = CNN().to(DEVICE)
    cnn.load_state_dict(torch.load("cnn_model.pt", map_location=DEVICE))
    cnn.eval()

    wav2vec_model = Model().to(DEVICE)
    wav2vec_model.load_state_dict(torch.load("wav2vec2_model.pt", map_location=DEVICE))
    wav2vec_model.eval()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    return cnn, wav2vec_model, processor

cnn, wav2vec_model, processor = load_models()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS
)


def load_audio_bytes(audio_bytes):
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] < NUM_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :NUM_SAMPLES]

    return waveform


def predict(waveform):
    # CNN
    mel = torch.log(mel_transform(waveform) + 1e-9)
    mel = mel.unsqueeze(0).to(DEVICE)
    p1 = torch.sigmoid(cnn(mel)).item()

    # wav2vec2
    inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    x = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        p2 = torch.sigmoid(wav2vec_model(x)).item()

    return (p1 + p2) / 2


uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["mp3", "wav", "opus", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    audio_bytes = uploaded_file.read()

    with st.spinner("Analyzing audio..."):
        waveform = load_audio_bytes(audio_bytes)
        score = predict(waveform)

    st.subheader("Result")

    if score > 0.5:
        st.error("AI GENERATED")
    else:
        st.success("REAL")

    st.write(f"**Confidence Score:** {score:.3f}")

    # Optional interpretation
    if score > 0.75:
        st.warning("High confidence of synthetic voice")
    elif score > 0.5:
        st.warning("Moderate confidence of synthetic voice")
    else:
        st.info("Likely natural human speech")
