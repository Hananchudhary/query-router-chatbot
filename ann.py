import pandas as pd
import numpy as np
import torch
import re

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("dataset.csv")

required_columns = [
    "clean_text",
    "time",
    "is_question",
    "query_length",
    "label"
]

df = df[required_columns].dropna()

# =========================================================
# PREPROCESSING
# =========================================================

STOPWORDS = {
    "the","a","an","is","are","to","of","in","on","for","and","or",
    "with","this","that","it","be","as","at","by","from"
}

def clean_sentence(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    return " ".join([w for w in words if w not in STOPWORDS])

def preprocess_dataframe(df):
    df = df.copy()

    df["clean_text"] = df["clean_text"].astype(str).apply(clean_sentence)
    df["label"] = df["label"].astype(int)
    df["is_question"] = df["is_question"].astype(int)
    df["query_length"] = df["query_length"].astype(int)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["hour"] = df["time"].dt.hour.fillna(0)
    df["dayofweek"] = df["time"].dt.dayofweek.fillna(0)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    return df

df = preprocess_dataframe(df)

# =========================================================
# TRAIN / TEST SPLIT (NEW)
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================================================
# VOCABULARY
# =========================================================

MAX_VOCAB_SIZE = 2000
MIN_WORD_FREQ = 2

word_freq = {}

for text in train_df["clean_text"]:
    for word in text.split():
        word_freq[word] = word_freq.get(word, 0) + 1

filtered_words = [
    (w, f) for w, f in word_freq.items()
    if f >= MIN_WORD_FREQ
]

filtered_words.sort(key=lambda x: x[1], reverse=True)
filtered_words = filtered_words[:MAX_VOCAB_SIZE]

vocab = {word: idx for idx, (word, _) in enumerate(filtered_words)}
print(f"Vocabulary Size: {len(vocab)}")

# =========================================================
# IDF
# =========================================================

num_docs = len(train_df)
vocab_size = len(vocab)

df_counts = np.zeros(vocab_size)

for text in train_df["clean_text"]:
    seen = set()
    for word in text.split():
        if word in vocab:
            idx = vocab[word]
            if idx not in seen:
                df_counts[idx] += 1
                seen.add(idx)

idf = np.log((1 + num_docs) / (1 + df_counts)) + 1

# =========================================================
# TF-IDF
# =========================================================

def vectorize_text(df):
    X = np.zeros((len(df), vocab_size))

    for i, text in enumerate(df["clean_text"]):
        words = text.split()
        tf_counts = {}

        for w in words:
            if w in vocab:
                idx = vocab[w]
                tf_counts[idx] = tf_counts.get(idx, 0) + 1

        for idx, count in tf_counts.items():
            tf = count / len(words)
            X[i, idx] = tf * idf[idx]

    return X

X_train_text = vectorize_text(train_df)
X_test_text = vectorize_text(test_df)

# =========================================================
# NUMERIC FEATURES
# =========================================================

features = ["is_question", "query_length", "hour", "dayofweek", "is_weekend"]

X_train_numeric = train_df[features].values
X_test_numeric = test_df[features].values

mean = X_train_numeric.mean(axis=0)
std = X_train_numeric.std(axis=0) + 1e-8

X_train_numeric = (X_train_numeric - mean) / std
X_test_numeric = (X_test_numeric - mean) / std

# =========================================================
# FINAL FEATURES
# =========================================================

X_train = np.hstack([X_train_text, X_train_numeric])
X_test = np.hstack([X_test_text, X_test_numeric])

y_train = train_df["label"].values
y_test = test_df["label"].values

# =========================================================
# TORCH TENSORS
# =========================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# =========================================================
# DATASET
# =========================================================

class QueryIntentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = QueryIntentDataset(X_train, y_train)
test_dataset = QueryIntentDataset(X_test, y_test)

# =========================================================
# DATALOADER
# =========================================================

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# MODEL
# =========================================================

class QueryIntentANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.network(x)

model = QueryIntentANN(X_train.shape[1]).to(device)
print(model)

# =========================================================
# LOSS + OPTIMIZER
# =========================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# =========================================================
# ACCURACY
# =========================================================

def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# =========================================================
# TRAIN STEP
# =========================================================

def train_step():
    model.train()
    total_loss, total_acc = 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        total_acc += accuracy_fn(y, preds)

    return total_loss / len(train_loader), total_acc / len(train_loader)

# =========================================================
# TEST STEP
# =========================================================

def test_step():
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_acc += accuracy_fn(y, preds)

    return total_loss / len(test_loader), total_acc / len(test_loader)

# =========================================================
# TRAINING LOOP
# =========================================================

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss, train_acc = train_step()
    test_loss, test_acc = test_step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# =========================================================
# FINAL EVALUATION
# =========================================================

model.eval()

with torch.inference_mode():
    logits = model(X_test.to(device))
    preds = torch.argmax(logits, dim=1)

    final_acc = accuracy_fn(y_test.to(device), preds)

print(f"\nFinal Test Accuracy: {final_acc:.2f}%")