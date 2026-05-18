!pip install gradio
import pandas as pd
import numpy as np
import torch
import re
import gradio as gr
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

df = pd.read_csv("dataset.csv")

required_columns = [
    "clean_text",
    "time",
    "is_question",
    "query_length",
    "label"
]

labels = ["General", "Code Generation","Code Issues","Educational","Lookup"]
df = df[required_columns].dropna()

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

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(0.8 * len(df))

train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

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

features = ["is_question", "query_length", "hour", "dayofweek", "is_weekend"]

X_train_numeric = train_df[features].values
X_test_numeric = test_df[features].values

mean = X_train_numeric.mean(axis=0)
std = X_train_numeric.std(axis=0) + 1e-8

X_train_numeric = (X_train_numeric - mean) / std
X_test_numeric = (X_test_numeric - mean) / std

X_train = np.hstack([X_train_text, X_train_numeric])
X_test = np.hstack([X_test_text, X_test_numeric])

y_train = train_df["label"].values
y_test = test_df["label"].values

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

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

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return (correct / len(y_pred)) * 100

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

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss, train_acc = train_step()
    test_loss, test_acc = test_step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

model.eval()

with torch.inference_mode():
    logits = model(X_test.to(device))
    preds = torch.argmax(logits, dim=1)

    final_acc = accuracy_fn(y_test.to(device), preds)
    cm = confusion_matrix(
        y_test.to(device).numpy(),
        preds.to(device).numpy()
    )
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix - Query Intent Classifier")
plt.show()
print(f"\nFinal Test Accuracy: {final_acc:.2f}%")

def get_numeric_features(text):
    now = datetime.now()

    hour = now.hour
    dayofweek = now.weekday()
    is_weekend = 1 if dayofweek in [5, 6] else 0

    return np.array([
        1 if "?" in text else 0,
        len(text.split()),
        hour,
        dayofweek,
        is_weekend
    ])

def vectorize(text):
    words = clean_sentence(text)

    x_text = np.zeros(vocab_size)
    tf_counts = {}

    for w in words:
        if w in vocab:
            idx = vocab[w]
            tf_counts[idx] = tf_counts.get(idx, 0) + 1

    for idx, count in tf_counts.items():
        tf = count / max(len(words), 1)
        x_text[idx] = tf * idf[idx]

    return x_text


def preprocess_input(text):
    x_text = vectorize(text)
    x_num = get_numeric_features(text)

    x_num = (x_num - mean) / (std + 1e-8)

    x = np.hstack([x_text, x_num])
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)


def predict(query):
    x = preprocess_input(query)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    return {
        "Predicted Class": labels[pred_class],
        "Confidence": float(np.max(probs))
    }

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query..."),
    outputs=gr.JSON(),
    title="🧠 Query Intent Classifier",
    description="Enter a query and the model predicts its intent class."
)

interface.launch()