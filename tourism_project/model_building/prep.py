# for data manipulation
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/mkrish2025/Tourism-Customer-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(df.columns[0], axis=1,inplace=True) #running number of rows
df.drop(columns=['CustomerID'], inplace=True)

# Correcting data gaps
df['Gender'] = df['Gender'].apply(lambda x: 'Female' if x == 'Fe Male' else x)
df['Occupation'] = df['Occupation'].apply(lambda x: 'Salaried' if x == 'Free Lancer' else x)

# Combining similar representation as Total no of visitors.
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

# Customer Interaction Data will not be required for prediction as this is updated post interaction.
# Will retain 'ProductPitched' alone as this will be required for pitching the product
df.drop(columns=['PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'], axis=1, inplace=True)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ['Age','MonthlyIncome']
categorical_features = ['Designation', 'OwnCar', 'Passport',
            'CityTier','MaritalStatus',
            'ProductPitched','Gender','Occupation','TypeofContact'
            ]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder="passthrough"
)

# Fit ONLY on training data
Xtrain_processed = preprocessor.fit_transform(Xtrain)
Xtest_processed = preprocessor.transform(Xtest)


feature_names = preprocessor.get_feature_names_out()

Xtrain_df = pd.DataFrame(
    Xtrain_processed,
    columns=feature_names
)

Xtest_df = pd.DataFrame(
    Xtest_processed,
    columns=feature_names
)


Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="mkrish2025/Tourism-Customer-Prediction",
        repo_type="dataset",
    )
