from transformers import BertTokenizer, BertForSequenceClassification
from time import sleep
import torch
import os
import pandas as pd #Used for reading Excel files
import docx  # For reading Word filess
import PyPDF2  # For reading PDF files
from create_model import train_and_create_model

folder_path = "backup"
to_be_deleted = os.path.join(folder_path, "delete")
trained_model = "fine_tuned_model"

# Create backup folder if it does not exist:
if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)

# Create delete folder inside the backup folder if it does not exist:
if not os.path.exists(to_be_deleted):
    os.makedirs(to_be_deleted, exist_ok=True)

# Create model if it does not exist
if not os.path.exists(trained_model):
    train_and_create_model()

# Function to read text data from Excel files
def read_excel_files(file_paths):
    text_data = []
    for file_path in file_paths:
        xls = pd.ExcelFile(file_path, engine='openpyxl')  # Use openpyxl engine
        for sheet_name in xls.sheet_names:
            pandas_read = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            for row in pandas_read.values:
                for cell in row:
                    text_data.append(str(cell))
        xls.close()
    return text_data

# Function to read text data from Word files
def read_word_files(file_paths):
    text_data = []
    for file_path in file_paths:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text_data.append(paragraph.text)
    return text_data

# Function to read text data from PDF files
def read_pdf_files(file_paths):
    text_data = []
    for file_path in file_paths:
        pdf_file = open(file_path, 'rb')
        reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text_data.append(page.extract_text())
        pdf_file.close()
    return text_data

# Function to list all supported files in a folder
def list_supported_files(folder_path):
    supported_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.xlsx', '.xls', '.docx', '.pdf')):
            supported_files.append(os.path.join(folder_path, file_name))
    return supported_files

# Load all supported files in the folder
supported_files = list_supported_files(folder_path)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # uncased = does not look differently at Apples and apples. They are the same

# Load the fine-tuned BERT model
model = BertForSequenceClassification.from_pretrained('fine_tuned_model')

def classify_files():
    predicted_labels_list = []  # Store the predicted labels for all files
    # Loop through each supported file and classify them
    for file_path in supported_files:
        text_data = []
        if file_path.endswith(('.xlsx', '.xls')):
            text_data = read_excel_files([file_path])
        elif file_path.endswith('.docx'):
            text_data = read_word_files([file_path])
        elif file_path.endswith('.pdf'):
            text_data = read_pdf_files([file_path])
        
        # Tokenize the text data using BERT tokenizer
        inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
        
        # Perform classification
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted labels
        predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
        predicted_labels_list.append(predicted_labels[0])  # Add predicted label to the list
        
        # Print predicted labels for the file
        print(f"File: {file_path} | Predicted label: {predicted_labels[0]}")
        
    return predicted_labels_list  # Return the list of predicted labels

sleep(2)

def move_files_to_delete_folder():
    print("Moving unnecessary files to the delete folder...")
    for label, file_path in zip(predicted_labels, supported_files):
        if label == 0:  # Non-finance label
            file_name = os.path.basename(file_path)
            destination = os.path.join(to_be_deleted, file_name)
            sleep(2)
            try:
                os.rename(file_path, destination)
                print(f"Moved file '{file_name}' to '{to_be_deleted}'.")
            except PermissionError:
                print(f"ERROR: File '{file_name}' is currently in use and cannot be moved.")
            except Exception as e:
                print(f"ERROR: Failed to move file '{file_name}': {e}")

predicted_labels = classify_files()
sleep(2)
move_files_to_delete_folder()
# Use the following code if you whish to delete the files instead
"""
def delete_files():
    print("Trying to delete uneeded files.....")
    for label, file_path in zip(predicted_labels, supported_files):
        if label == 0:  # Non-finance label
            file_name = os.path.basename(file_path)
            sleep(2)
            try:
                os.remove(file_path)
                print(f"Deleted file '{file_name}'.")
            except PermissionError:
                print(f"ERROR: File '{file_name}' is currently in use and cannot be deleted.")
            except Exception as e:
                print(f"ERROR: Failed to delete file '{file_name}': {e}")
"""
#delete_files()