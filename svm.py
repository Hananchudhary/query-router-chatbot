import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/hanan/Projects/works/query-router/dataset.csv")

df = df.dropna(subset=["clean_text", "label"])

df["clean_text"] = df["clean_text"].astype(str)
df["label"] = df["label"].astype(int)

df["is_question"] = df["is_question"].astype(int)

df["time"] = pd.to_datetime(df["time"], errors="coerce")

df["hour"] = df["time"].dt.hour.fillna(0)
df["dayofweek"] = df["time"].dt.dayofweek.fillna(0)
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

vocab = {}
for text in df["clean_text"]:
    for w in text.lower().split():
        if w not in vocab:
            vocab[w] = len(vocab)

vocab_size = len(vocab)
num_docs = len(df)

X_text = np.zeros((num_docs, vocab_size))

for i, text in enumerate(df["clean_text"]):
    words = text.lower().split()

    for w in words:
        if w in vocab:
            X_text[i, vocab[w]] += 1

X_text = X_text / (np.sum(X_text, axis=1, keepdims=True) + 1e-8)

df_count = np.sum(X_text > 0, axis=0)
idf = np.log((1 + num_docs) / (1 + df_count)) + 1

X_text = X_text * idf

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

X = np.hstack([X_text, X_numeric])
y = df["label"].values

num_classes = len(np.unique(y))

idx = np.arange(len(X))
np.random.shuffle(idx)

split = int(0.8 * len(X))

train_idx = idx[:split]
test_idx = idx[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

class BinarySVM:

    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=300):

        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.w = None
        self.b = 0
        self.loss_history = []

    def compute_loss(self, X, y):

        distances = 1 - y * (np.dot(X, self.w) - self.b)
        hinge = np.maximum(0, distances)

        return (
            self.lambda_param * np.dot(self.w, self.w)
            + np.mean(hinge)
        )

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.n_iters):

            for i, x_i in enumerate(X):

                condition = y[i] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y[i] * x_i
                    db = y[i]

                self.w -= self.lr * dw
                self.b -= self.lr * db

            if epoch % 20 == 0:
                self.loss_history.append(self.compute_loss(X, y))

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

class MultiClassSVM:

    def __init__(self, n_classes, lr=0.01, lambda_param=0.01, n_iters=300):

        self.n_classes = n_classes

        self.models = [
            BinarySVM(lr, lambda_param, n_iters)
            for _ in range(n_classes)
        ]

    def fit(self, X, y):

        for c in range(self.n_classes):

            print(f"Training class {c} vs rest...")

            y_binary = np.where(y == c, 1, -1)

            self.models[c].fit(X, y_binary)

    def predict(self, X):

        scores = np.zeros((X.shape[0], self.n_classes))

        for c, model in enumerate(self.models):

            scores[:, c] = model.decision_function(X)

        return np.argmax(scores, axis=1)

model = MultiClassSVM(
    n_classes=num_classes,
    lr=0.01,
    lambda_param=0.01,
    n_iters=200
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100

print(f"Test Accuracy: {accuracy:.2f}%")
