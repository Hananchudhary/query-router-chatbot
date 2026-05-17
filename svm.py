import csv
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
queries, labels = [], []
with open('dataset.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        queries.append(row[1])  # clean_text
        labels.append(int(row[5]))

X_train, X_test, y_train, y_test = train_test_split(
    queries, labels, test_size=0.2, random_state=42, stratify=labels
)

# Pipeline with TF-IDF + Linear SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.85, min_df=2, ngram_range=(1, 2), sublinear_tf=True
    )),
    ('clf', LinearSVC(class_weight='balanced', dual='auto', max_iter=10000))
])

# Grid search over C
param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best C: {grid.best_params_['clf__C']}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print()

y_pred = grid.predict(X_test)
acc = np.mean(y_pred == y_test)
print(f"Test accuracy: {acc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=[
    'chit-chat', 'coding-task', 'error', 'educational', 'conceptual-Q&A'
]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    grid, X_test, y_test, ax=ax,
    display_labels=['chat', 'code', 'error', 'edu', 'concept'],
    cmap='Blues', values_format='d'
)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Saved confusion_matrix.png")
