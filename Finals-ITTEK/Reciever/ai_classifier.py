from transformers import BertTokenizer, BertForSequenceClassification
from time import sleep
import glob
import torch
import os
import shutil # Used for moving files
import pandas as pd

# Folder containing Excel files
folder_path = "backup"
delete_folder = "backup/delete"
files_to_move = []

# Creates delete folder if it does not exist
if not os.path.exists(delete_folder):
    os.makedirs(delete_folder)

# Function to read text data from Excel files
def read_excel_files(file_paths):
    text_data = []
    for file_path in file_paths:
        #xls = pd.ExcelFile(file_path)
        xls = pd.ExcelFile(file_path, engine='openpyxl')  # Use openpyxl engine
        for sheet_name in xls.sheet_names:
            pandas_read = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            for row in pandas_read.values:
                for cell in row:
                    text_data.append(str(cell))
        xls.close
    return text_data

# Function to list all Excel files in a folder
def list_excel_files(folder_path):
    excel_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            excel_files.append(os.path.join(folder_path, file_name))
    return excel_files

# Load all Excel files in the folder
excel_files = list_excel_files(folder_path)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the fine-tuned BERT model
model = BertForSequenceClassification.from_pretrained('fine_tuned_model')


def classify_files():
    predicted_labels_list = []  # Store the predicted labels for all files
    # Loop through each Excel file and classify them
    for excel_file in excel_files:
        # Read text data from Excel file
        text_data = read_excel_files([excel_file])
        
        # Tokenize the text data using BERT tokenizer
        inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
        
        # Perform classification
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted labels
        predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
        predicted_labels_list.append(predicted_labels[0])  # Add predicted label to the list
        
        # Print predicted labels for the Excel file
        print(f"File: {excel_file} | Predicted label: {predicted_labels[0]}")
        
    return predicted_labels_list  # Return the list of predicted labels
sleep(2)

def delete_files():
    print("Trying to delete uneeded files.....")
    for label, file_path in zip(predicted_labels, excel_files):
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
    

predicted_labels= classify_files()
sleep(2)
delete_files()



"""
def classify_files():
    # Loop through each Excel file and classify them
    for excel_file in excel_files:
        # Read text data from Excel file
        text_data = read_excel_files([excel_file])
        
        # Tokenize the text data using BERT tokenizer
        inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
        
        # Perform classification
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted labels
        predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
        
        # Print predicted labels for the Excel file
        print(f"File: {excel_file} | Predicted label: {predicted_labels[0]}")
        values = predicted_labels[0] 
        return values


def move_files_to_delete():
# LAV EN FUNCTION AF DET FØRSTE SOM RETUNERE, DEREFTER BRUG DET RETUNEREDE I DENNE DEN AF KODEN
# Move the file to the "delete" folder if predicted as non-finance
    if predicted_labels[0] == 0:
        file_name = os.path.basename(excel_files)
        new_file_path = os.path.join(delete_folder, file_name)
        os.replace(excel_files, new_file_path)
        print(f"Moved file '{file_name}' to '{delete_folder}' folder.")

classify_files()
move_files_to_delete(values)
"""