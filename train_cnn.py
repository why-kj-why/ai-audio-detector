import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import io


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 4
BATCH_SIZE = 16
EPOCHS = 10
THRESHOLD = 0.5
LR = 1e-3
N_MELS = 64
TEST_SIZE = 0.25


dataset = load_dataset("garystafford/deepfake-audio-detection")
df = dataset["train"].to_pandas()

train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=42
)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS
)


def preprocess(audio_bytes):
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] < NUM_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :NUM_SAMPLES]

    mel = torch.log(mel_transform(waveform) + 1e-9)
    return mel


class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = preprocess(row["audio"]["bytes"])
        label = row["label"]
        return mel, torch.tensor(label, dtype=torch.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.net(x))


def train():
    train_loader = DataLoader(AudioDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)

            loss = crit(model(x), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "cnn_model.pt")
    return model


def evaluate(model):
    loader = DataLoader(AudioDataset(df), batch_size=BATCH_SIZE)

    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            probs = torch.sigmoid(model(x)).squeeze()
            preds = (probs > THRESHOLD).int().cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    recall = tp / (tp + fn)

    print("\n=== CNN Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}, Recall: {recall:.4f}")


if __name__ == "__main__":
    model = train()
    evaluate(model)
