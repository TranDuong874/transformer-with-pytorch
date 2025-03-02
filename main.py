import torch
from transformers import AutoTokenizer
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from src.transformer import TransformerSentiment
import torch.optim as optim


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=250, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        tokens = tokenize_text(text)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


df = pd.read_csv("data/data2.csv")
dataset = SentimentDataset(df["text"].tolist(), df["label"].tolist())

from torch.utils.data import DataLoader

batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in train_loader:
    print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["label"].shape)
    break

# After creating dataset
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerSentiment(vocab_size=len(tokenizer), d_model=512, num_heads=8, d_ff=2048, n_layers=6, num_classes=3).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

print(f"Model device: {next(model.parameters()).device}")
# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # print(f"Input device: {input_ids.device}")
        optimizer.zero_grad()
        outputs, loss = model(input_ids, input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")