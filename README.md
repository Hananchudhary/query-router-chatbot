# Query Router Chatbot

This project implements a chatbot that routes user queries to the appropriate category using machine learning models. It includes two main approaches: an Artificial Neural Network (ANN) and a Support Vector Machine (SVM).

## Features

*   Query intent classification
*   ANN model implementation using PyTorch
*   SVM model implementation
*   Web-based interface for predictions using Gradio

## Project Structure

*   `ann.py`: Implements the ANN model for query intent classification.
*   `svm.py`: Implements the SVM model for query intent classification.
*   `dataset.csv`: The dataset used for training and evaluating the models.

## Dataset Collection

* `General Class`: From whatsapp and Instagram Chats
* `Other` : From Stack Overflow API and from exporting chats from different     LLMs

## Data Preprocessing

* Normalized time format
* Added more features for better prediction using scripting

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd query-router
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the models:**
    *   To run the ANN model:
        ```bash
        python ann.py
        ```
    *   To run the SVM model:
        ```bash
        python svm.py
        ```

## About the models

### 1. ANN (Artificial Neural Network)

The ANN model is implemented using PyTorch and consists of:

- **Architecture**: A feedforward neural network with two hidden layers (64 and 32 neurons) with ReLU activation and dropout (0.5) for regularization
- **Input Features**: 
  - TF-IDF vectorized text features (up to 2000 vocabulary words)
  - Numeric features: is_question, query_length, hour, dayofweek, is_weekend
- **Output**: 5-class classification (General, Code Generation, Code Issues, Educational, Lookup)
- **Training**: Adam optimizer with learning rate 0.001, CrossEntropyLoss, trained for 10 epochs
- **Batch Size**: 16
- **Device**: Automatically uses GPU if available (CUDA), otherwise CPU

### 2. SVM (Support Vector Machine)

The SVM model is a custom implementation using a One-vs-Rest approach for multi-class classification:

- **Architecture**: Multiple binary SVM classifiers (one per class) using hinge loss
- **Input Features**: Same as ANN (TF-IDF text + normalized numeric features)
- **Output**: 5-class classification using argmax of decision function scores
- **Training**: Custom gradient descent with L2 regularization (lambda=0.01), learning rate 0.01, 200 iterations
- **Approach**: Each binary SVM learns to distinguish one class from all others

## Usage

Once the application is running, you can interact with the chatbot through the Gradio web interface. Enter your query, and the model will predict its intent.