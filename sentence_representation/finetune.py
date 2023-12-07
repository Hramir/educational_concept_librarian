import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class BertRegressionHead(nn.Module):
    def __init__(self, config):
        super(BertRegressionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, 1)
        
    def forward(self, x):
        x = self.dense(x)
        return x

def train_model(df):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    model.classifier = BertRegressionHead(model.config)

    # Prepare data
    texts = df["sentence_embedding"].tolist()
    df['view_count'] = np.log10(df['view_count'])
    df['view_count_standardized'] = (df["view_count"]-df["view_count"].mean())/df["view_count"].std()
    targets = (df["view_count_standardized"]).astype(float).tolist()

    # Split the data into training, validation, and test sets
    texts_train, texts_temp, targets_train, targets_temp = train_test_split(texts, targets, test_size=0.2, random_state=20)
    texts_val, texts_test, targets_val, targets_test = train_test_split(texts_temp, targets_temp, test_size=0.5, random_state=20)

    # Tokenize and prepare data
    tokenized_train = tokenizer(texts_train, padding=True, truncation=True, return_tensors="pt")
    tokenized_val = tokenizer(texts_val, padding=True, truncation=True, return_tensors="pt")
    tokenized_test = tokenizer(texts_test, padding=True, truncation=True, return_tensors="pt")

    train_dataset = TensorDataset(tokenized_train["input_ids"], tokenized_train["attention_mask"], torch.tensor(targets_train))
    val_dataset = TensorDataset(tokenized_val["input_ids"], tokenized_val["attention_mask"], torch.tensor(targets_val))
    test_dataset = TensorDataset(tokenized_test["input_ids"], tokenized_test["attention_mask"], torch.tensor(targets_test))

    # Define loss function, optimizer, and evaluation metric
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 15
    batch_size = 16

    for epoch in range(num_epochs):
        model.train()
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = []
            for val_batch in DataLoader(val_dataset, batch_size=batch_size):
                val_input_ids, val_attention_mask, val_labels = val_batch
                val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                val_predictions.extend(val_outputs.logits.squeeze().tolist())
            val_loss = criterion(torch.tensor(val_predictions), torch.tensor(targets_val))
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss.item()}')

    # Test
    model.eval()
    with torch.no_grad():
        test_predictions = []
        for test_batch in DataLoader(test_dataset, batch_size=batch_size):
            test_input_ids, test_attention_mask, test_labels = test_batch
            test_outputs = model(test_input_ids, attention_mask=test_attention_mask)
            test_predictions.extend(test_outputs.logits.squeeze().tolist())

    # Calculate test metrics
    test_loss = mean_squared_error(test_predictions, targets_test)
    print(f'Test Loss: {test_loss}')
    print(targets_test)
    print(test_predictions)

if __name__ == "__main__":
    df = pd.read_csv("sentence_embed.csv")
    train_model(df[["view_count", "sentence_embedding"]].dropna())