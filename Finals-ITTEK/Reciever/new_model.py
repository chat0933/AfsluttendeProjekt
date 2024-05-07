from time import sleep
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from list_of_words import finance_related_text, non_finance_related_text

positives = finance_related_text
negatives = non_finance_related_text
# Folder containing Excel files
folder_path = "Files" 
# Combine the positive and negative examples i have created
pos_and_neg_values = positives + negatives
training_labels = ['finance-related'] * len(positives) + ['non-finance-related'] * len(negatives)


"""
Function to read text data from Excel files
It reads one word at a time from an excel file
when one word has been read it adds the next word to the list
When every word has been read it goes on to the next excel file
"""
def read_excel_files(file_paths):
    text_data = []
    for file_path in file_paths:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            pandas_read = pd.read_excel(xls, sheet_name=sheet_name, header=None)  # Reading each sheet without considering headers
            for row in pandas_read.values:
                for cell in row:
                    text_data.append(str(cell))
    return text_data

# Function to list all Excel files in a folder
# Adds every excel file to a list that is used later
def list_excel_files(folder_path):
    excel_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            excel_files.append(os.path.join(folder_path, file_name))
            #print(excel_files)
    return excel_files

# List all Excel files in the folder
excel_files = list_excel_files(folder_path)

# Feature extraction using TF-IDF
# It takes every word and makes TF-IDF values out of pos and neg values
# Later they get compared with the excel files
# The Vectorizer calculates the TF-IDF
vectorizer = TfidfVectorizer()
text_tfidf = vectorizer.fit_transform(pos_and_neg_values)
#print(text_tfidf)

"""
Train a logistic regression classifier
Here the TF-IDF values goes into the clasifier
The LogisticRegression model which is an linear regression model uses the vectorizer TF
"""
classifier = LogisticRegression()
classifier.fit(text_tfidf[:len(pos_and_neg_values)], training_labels)

""" 
Kig på youtube videoer omkring sklearn, logistic regression
Skriv process ned når jeg ændrer ting på modellen, så jeg kan se historikken på programmet (a-b-c)
Gør listen af ord / sætninger længere. Kig måske hellere på LLama3 (META har lavet den)
huggingface.com? (kig på den fordi folk ligger deres modeller op)
"""


# Loop through each Excel file and check them one at a time
for excel_file in excel_files:
    # Read text data from Excel file
    text_data = read_excel_files([excel_file])

    # Feature extraction using TF-IDF
    tfidf_features_combined = vectorizer.transform(pos_and_neg_values + text_data)
    #sleep(2)

    # Predict finance-related or non-finance-related for the text data from all the Excel file
    predicted_labels = classifier.predict(tfidf_features_combined[len(pos_and_neg_values):])

    # Print predicted labels for the Excel file
    print(f"File: {excel_file} | Predicted label: {predicted_labels[0]}")