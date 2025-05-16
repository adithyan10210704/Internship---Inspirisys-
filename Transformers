#Transformers for text classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import math

# Hyperparameters
vocab_size = 30522  # BERT tokenizer vocab size
d_model = 128  # Embedding size
nhead = 4  # Number of attention heads
num_layers = 2  # transformer layers
dff = 512  # Feed-forward network size
max_len = 128  # Maximum sequence length
num_classes = 2  # Binary classification 
batch_size = 32
epochs = 3
learning_rate = 1e-4

# Load IMDb dataset , tokenizer
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Data preprocessing
def preprocess_data(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

train_data = dataset["train"].map(preprocess_data, batched=True)
test_data = dataset["test"].map(preprocess_data, batched=True)

# Convert to PyTorch tensors
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return input_ids, labels

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

# Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dff, max_len, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dff, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Pooling over sequence
        x = self.fc(x)
        return x

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size, d_model, nhead, num_layers, dff, max_len, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training 
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    train_acc = train_correct / train_total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

# Evaluation
model.eval()
test_correct = 0
test_total = 0
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# Examples
print("\nExample Predictions (First 5):")
for i in range(5):
    text = tokenizer.decode(train_data[i]["input_ids"], skip_special_tokens=True)[:50] + "..."
    pred = "Positive" if predictions[i] == 1 else "Negative"
    true = "Positive" if true_labels[i] == 1 else "Negative"
    print(f"Text: {text}")
    print(f"Predicted: {pred}, True: {true}")
