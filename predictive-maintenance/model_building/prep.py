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

# But we need to process the data properly.
# change the column names into small case and replace space with hyphen.
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

#verify if columns are renamed correctly and datatypes.
df.dtypes

#Upload the resulting train and test datasets back to the Hugging Face data space
X = df.drop("engine_condition", axis=1)
y = df["engine_condition"]

# use stratify to split the dataset evenly in train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv splitting is successful \n")

#Upload the resulting train and test datasets back to the Hugging Face data space
api = HfApi(token=os.getenv("HF_TOKEN"))

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="PratzPrathibha/predictive-maintenance-data",
        repo_type="dataset",
    )
