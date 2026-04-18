import torch
import torchaudio
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import Wav2Vec2Processor
from train_v1 import CNN
from train_v2 import Model


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 4
N_MELS = 64

cnn = CNN().to(DEVICE)
cnn.load_state_dict(torch.load("cnn_model.pt"))
cnn.eval()

wav2vec_model = Model().to(DEVICE)
wav2vec_model.load_state_dict(torch.load("wav2vec2_model.pt"))
wav2vec_model.eval()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS
)


def load_audio(path):
    waveform, sr = torchaudio.load(path)

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


ROOT_DIR = "./test-set"

label_map = {
    "real": 0,
    "synthetic": 1
}

results = []

for cls in os.listdir(ROOT_DIR):
    class_path = os.path.join(ROOT_DIR, cls)
    if not os.path.isdir(class_path):
        continue

    label = label_map[cls]

    for file in os.listdir(class_path):
        if file.startswith("."):
            continue

        if not file.lower().endswith((".mp3", ".wav", ".opus", ".flac")):
            continue

        path = os.path.join(class_path, file)

        try:
            waveform = load_audio(path)
            score = predict(waveform)
            pred = 1 if score > 0.5 else 0

            # filename without extension
            filename = os.path.splitext(file)[0]

            results.append((filename, label, pred, score))

            print(f"{file}")
            print(f"  True: {label} | Pred: {pred} | Score: {score:.3f}")
            print("-"*40)

        except Exception as e:
            print(f"Skipping {file}: {e}")



y_true = [r[1] for r in results]
y_pred = [r[2] for r in results]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
recall = tp / (tp + fn)

print("\n=== Ensemble Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}, Recall: {recall:.4f}")


def label_to_text(label):
    return "REAL" if label == 0 else "SYNTHETIC"

csv_data = [
    {
        "filename": r[0],
        "actual_value": label_to_text(r[1]),
        "predicted_value": label_to_text(r[2]),
        "confidence_score": round(r[3], 3)
    }
    for r in results
]

df = pd.DataFrame(csv_data)
df.to_csv("test_results.csv", index=False)
