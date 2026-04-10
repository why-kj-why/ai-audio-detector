import torch
import torchaudio
import torch.nn as nn
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import io


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
TEST_SIZE = 0.25

dataset = load_dataset("garystafford/deepfake-audio-detection")
df = dataset["train"].to_pandas()

train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"], random_state=42)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)

for p in wav2vec.parameters():
    p.requires_grad = False

for p in wav2vec.encoder.layers[-2:].parameters():
    p.requires_grad = True


class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(io.BytesIO(row["audio"]["bytes"]))

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        waveform = waveform.mean(dim=0)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

        return inputs.input_values.squeeze(0), row["label"]


def collate_fn(batch):
    x = [b[0] for b in batch]
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    return x, y


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out = wav2vec(x.to(DEVICE)).last_hidden_state
        pooled = out.mean(dim=1)
        return self.fc(pooled)


def train():
    loader = DataLoader(AudioDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = Model().to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)

            loss = crit(model(x), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "wav2vec2_model.pt")
    return model


def evaluate(model):
    loader = DataLoader(AudioDataset(df), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            probs = torch.sigmoid(model(x)).squeeze()
            preds = (probs > 0.5).int().cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    recall = tp / (tp + fn)

    print("\n=== Wav2Vec2 Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}, Recall: {recall:.4f}")


if __name__ == "__main__":
    model = train()
    evaluate(model)