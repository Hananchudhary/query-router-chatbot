import pandas as pd
import numpy as np
import torch
import random
import time

from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader
)

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

df = pd.read_csv("dataset.csv")

df = df[
    [
        "clean_text",
        "time",
        "is_question",
        "query_length",
        "label"
    ]
]

df = df.dropna()

df["clean_text"] = df["clean_text"].astype(str)
df["label"] = df["label"].astype(int)
df["is_question"] = df["is_question"].astype(int)
df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)
df["hour"] = (
    df["time"]
    .dt.hour
    .fillna(0)
)
df["dayofweek"] = (
    df["time"]
    .dt.dayofweek
    .fillna(0)
)
df["is_weekend"] = (
    df["dayofweek"]
    .isin([5, 6])
    .astype(int)
)

vocab = {}
for text in df["clean_text"]:

    words = text.lower().split()
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)

print(f"Vocabulary Size: {len(vocab)}")

num_docs = len(df)

vocab_size = len(vocab)

df_counts = np.zeros(vocab_size)

for text in df["clean_text"]:
    seen_words = set()
    for word in text.lower().split():
        if word in vocab:
            idx = vocab[word]
            if idx not in seen_words:
                df_counts[idx] += 1
                seen_words.add(idx)

idf = np.log((1 + num_docs) / (1 + df_counts)) + 1
X_text = np.zeros((num_docs, vocab_size))

for i, text in enumerate(df["clean_text"]):
    words = text.lower().split()
    tf_counts = {}
    for word in words:
        if word in vocab:
            idx = vocab[word]
            tf_counts[idx] = (tf_counts.get(idx, 0) + 1)
    for idx, count in tf_counts.items():
        tf = count / len(words)
        X_text[i, idx] = tf * idf[idx]

X_numeric = np.column_stack([
    df["is_question"].values,
    df["query_length"].values,
    df["hour"].values,
    df["dayofweek"].values,
    df["is_weekend"].values
])

mean = X_numeric.mean(axis=0)
std = X_numeric.std(axis=0) + 1e-8
X_numeric = (X_numeric - mean) / std

X = np.hstack([
    X_text,
    X_numeric
])
y = df["label"].values

dataset_size = len(X)
indices = np.arange(dataset_size)
np.random.shuffle(indices)
train_size = int(0.8 * dataset_size)
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)
X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)
y_train = torch.tensor(
    y_train,
    dtype=torch.long
)
y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

class QueryIntentDataset(Dataset):

    def __init__(self, X, y):

        self.X = X

        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

BATCH_SIZE = 32

train_dataset = QueryIntentDataset(
    X_train,
    y_train
)

test_dataset = QueryIntentDataset(
    X_test,
    y_test
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class QueryIntentANN(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_units,
        output_features
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                hidden_units
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_units,
                hidden_units // 2
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_units // 2,
                output_features
            )
        )

    def forward(self, x):

        return self.network(x)

input_features = X_train.shape[1]

model = QueryIntentANN(
    input_features=input_features,
    hidden_units=256,
    output_features=5
).to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

def accuracy_fn(y_true, y_pred):

    correct = (
        y_true == y_pred
    ).sum().item()

    acc = (
        correct / len(y_pred)
    ) * 100

    return acc

def train_step(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device
):

    model.train()

    train_loss = 0

    train_acc = 0

    for X, y in dataloader:

        X = X.to(device)

        y = y.to(device)

        # Forward Pass
        y_logits = model(X)

        # Loss
        loss = loss_fn(
            y_logits,
            y
        )

        train_loss += loss.item()

        # Predictions
        y_pred = torch.argmax(
            y_logits,
            dim=1
        )

        train_acc += accuracy_fn(
            y,
            y_pred
        )

        # Zero Gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

    train_loss /= len(dataloader)

    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(
    model,
    dataloader,
    loss_fn,
    device
):

    model.eval()

    test_loss = 0

    test_acc = 0

    with torch.inference_mode():

        for X, y in dataloader:

            X = X.to(device)

            y = y.to(device)

            # Forward Pass
            test_logits = model(X)

            # Loss
            loss = loss_fn(
                test_logits,
                y
            )

            test_loss += loss.item()

            # Predictions
            test_pred = torch.argmax(
                test_logits,
                dim=1
            )

            test_acc += accuracy_fn(
                y,
                test_pred
            )

    test_loss /= len(dataloader)

    test_acc /= len(dataloader)

    return test_loss, test_acc

EPOCHS = 20

results = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

train_time_start = time.time()

for epoch in tqdm(range(EPOCHS)):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    test_loss, test_acc = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    results["train_loss"].append(train_loss)

    results["train_acc"].append(train_acc)

    results["test_loss"].append(test_loss)

    results["test_acc"].append(test_acc)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.2f}%"
    )

train_time_end = time.time()

print(
    f"\nTraining Time: "
    f"{train_time_end - train_time_start:.2f} seconds"
)

model.eval()

with torch.inference_mode():

    X_test_device = X_test.to(device)

    y_logits = model(X_test_device)

    y_pred = torch.argmax(
        y_logits,
        dim=1
    )

    final_acc = accuracy_fn(
        y_test.to(device),
        y_pred
    )

print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
