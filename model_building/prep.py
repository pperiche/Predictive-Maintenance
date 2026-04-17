# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
# for loading dataset from huggingface
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("PratzPrathibha/predictive-maintenance-data")
df = dataset["train"].to_pandas()

print("Dataset loaded successfully.... \n")

# reading the data and print head
pd.set_option('display.max_columns', None)
print(df.head())
print("Get the shape of data")
# print the number of rows and number of columns separately
print("Rows:    ", df.shape[0])
print("Columns: ", df.shape[1])
