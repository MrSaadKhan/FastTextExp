import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset

# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_bert_model(training_data, vector_size=100):
    # Initialize BERT tokenizer and configuration
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=vector_size)  # Setting hidden size
    model = BertForMaskedLM(config)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    # Prepare your dataset
    train_dataset = MyDataset(training_data)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Return the trained model
    return model

# Example usage:
# training_data = ["Your training sentences go here"]
# trained_model = train_bert_model(training_data, vector_size=100)
