from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import os
from list_of_words import finance_related_text, non_finance_related_text

# Combine finance-related and non-finance-related text data
pos_and_negative_values = finance_related_text + non_finance_related_text

# Create labels for the text data aka 1 (positive) and 0 (negative)
labels = [1] * len(finance_related_text) + [0] * len(non_finance_related_text) #Output = 1 and 0

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(pos_and_negative_values, labels, test_size=0.2, random_state=42) 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #uncased = converts everything to lowercase 

# Tokenize the text data
train_encodings = tokenizer(train_texts, truncation=True, padding=True) # padding adds 0, att_mask ? is 1 / 0. input_ids = num token 
test_encodings = tokenizer(test_texts, truncation=True, padding=True) #ditto

# Convert labels to tensors
train_labels = torch.tensor(train_labels) #Effektivt til computation (GPU)
test_labels = torch.tensor(test_labels)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) #AdamW adjusts weights, lr 0.000005, small adjustment for finetuning

#Create the training and testing datasets. Zip = samler
train_data = [{'input_ids': torch.tensor(encoding), 'attention_mask': torch.tensor(attention_mask), 'labels': label}
              for encoding, attention_mask, label in zip(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)]

test_data = [{'input_ids': torch.tensor(encoding), 'attention_mask': torch.tensor(attention_mask), 'labels': label}
             for encoding, attention_mask, label in zip(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)]

# Create DataLoader for training and testing data, batches. convert to tensor
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True) # shuffle is fine
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False) # Dont shuffle


def train_and_create_model():
    # Training loop
    model.train()
    for epoch in range(3):  # Amount of times the model gets trained / go through the data set
        for batch in train_dataloader:
            optimizer.zero_grad() # Clears gradients
            input_ids = batch['input_ids'] # numeral tokens
            attention_mask = batch['attention_mask'] #What is padding
            labels = batch['labels'] # 0 or 1
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) #process input, predict / sammenlign labels
            loss = outputs.loss # Incorrect predictions / how bad
            loss.backward() #calculates how each weight should be adjusted 
            optimizer.step() # ueses lr and gradients to optimize

    # Evaluation
    model.eval()
    predictions = []
    true_labels = []
    for batch in test_dataloader:
        with torch.no_grad(): # Gradients anvendes ikke, context manager 
            input_ids = batch['input_ids'] # Numeral tokens
            attention_mask = batch['attention_mask'] # What is padding
            labels = batch['labels'] # 0 or 1
            outputs = model(input_ids, attention_mask=attention_mask) #passed to model
            logits = outputs.logits #raw scores extraced
            predictions.extend(logits.argmax(dim=-1).cpu().numpy()) #Predicted labels, 1 dim =[1,2,3], argmax converts raw score
            true_labels.extend(labels.cpu().numpy()) # True labels, converts to numpy array

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) # Preiction / True labels = %
    print("Accuracy:", accuracy)

    # Save the trained model
    output_dir = "fine_tuned_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

#train_and_create_model()



"""
Use later if you want to develop the program more Line 34 -52
# Convert tokenized encodings to PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)

        
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)
"""
