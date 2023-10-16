#!/usr/bin/env python
# coding: utf-8

# In[15]:


# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
# import sklearn
# import sklearn
# import imblearn
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as imbalanced_Pipeline
# from sklearn.model_selection import train_test_split
# import random
# import numpy as np

#  # for model persistence

# # Load your pre-trained Gradient Boosting model
# model = joblib.load('gb_classifier.pkl')
# # preprocess_function = joblib.load("preprocess_data_function.pkl")
# # pipeline_function = joblib.load("pipeline_and_function.pkl")

# # Load the CSV file
# df = pd.read_csv('Latest_epl_data.csv')

# # Apply preprocessing to the loaded data
#     # Convert 'Date' and 'Time_x' to datetime
# df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time_x'])
# df['Time'] = pd.to_datetime(df['Time_x'], format='%H:%M').dt.time

#     # Fill missing values in 'Attendance' with seasonal median
# df['Season'] = df['Date'].dt.year
# seasonal_median = df.groupby('Season')['Attendance'].transform('median')
# df['Attendance'].fillna(seasonal_median, inplace=True)

#     # Fill missing values in 'Dist' with overall median
# median_dist = df['Dist'].median()
# df['Dist'].fillna(median_dist, inplace=True)

#     # Drop unnecessary columns
# columns_to_drop = ['Time_x', 'Comp', 'xGA', 'xG', 'Captain', 'Referee', 'Match Report', 'Notes', 'Cmp.1', 'Cmp.2', 'Cmp.3', 'Att.1', 'Att.2',
#                        'Att.3', 'Cmp%', 'Cmp%.1', 'Cmp%.2', 'Cmp%.3', 'Cmp.4', 'Cmp.5', 'Cmp.6', 'Cmp.7', 'Time_y']
# df.drop(columns=columns_to_drop, axis=1, inplace=True)
# class MissingDict(dict):
#     __missing__ = lambda self, key: key

# map_values = {"Brighton and Hove Albion ": "Brighton", "Manchester United ": "Manchester Utd", "Newcastle United ": "Newcastle Utd", "Tottenham Hotspur ": "Tottenham", "West Ham United ": "West Ham",
#               "Wolverhampton Wanderers ": "Wolves",'Newcastle United ': 'Newcastle Utd', 'West Ham United ': 'West Ham', 'Sheffield United ' : 'Sheffield Utd' }
# mapping = MissingDict(**map_values)
# df["Team"] = df["Team"].map(mapping)

#     # Map 'W', 'D', 'L' to numeric values in the 'Result' column
# result_mapping = {'W': 1, 'D': 0, 'L': 0}
# df['Result'] = df['Result'].map(result_mapping)

#     # Feature engineering
# df['Cmp_Poss_Ratio'] = df['Cmp'] / df['Poss']
# df['Cmp_PrgDist_Ratio'] = df['Cmp'] / df['PrgDist']

#     # Drop specified features
# features_to_drop = ['TotDist', 'Att', 'Poss', 'PrgDist']
# df.drop(features_to_drop, axis=1, inplace=True)

#     # Encode categorical features
# df["Venue"] = df["Venue"].astype("category").cat.codes
# df["opp_encoded"] = df["Opponent"].astype("category").cat.codes
# df["team_encoded"] = df["Team"].astype("category").cat.codes

#     # Convert 'Time' to whole number
# df['hour'] = df['Time'].astype(str).str.extract(r'(\d+)').astype(int)

#     # Map days of the week to code
# df["day_encoded"] = df["Date"].dt.dayofweek

#     # Convert 'Result' column from strings to integers
# df["Target"] = (df["Result"] == "W").astype("int")

#     # Encode 'Formation' column
# df["formation_encoded"] = df["Formation"].astype("category").cat.codes

#     # Create seasons
# df['Date'] = pd.to_datetime(df['Date'])

#     # Define a function to determine the season
# def get_season(date):
#     if date.month >= 8:
#         return date.year + 1
#     else:
#         return date.year

#     # Apply the function to create a new 'Season' column
# df['Season'] = df['Date'].apply(get_season)

#     # Convert 'Round' column to integer codes using label encoding
# df['Round_encoded'] = df['Round'].astype('category').cat.codes

#     # Drop encoded categorical features
# df.drop(columns=['Round', 'Day', 'Opponent', 'Formation', 'Team', 'Time', 'Date'], inplace=True)
#     # Define the numerical columns (excluding 'Formation')
# numerical_columns = ['GF', 'GA', 'Attendance', 'Sh', 'SoT', 'FK',
#                      'Dist', 'PK', 'PKatt', 'Cmp', 'Cmp_Poss_Ratio', 'Cmp_PrgDist_Ratio', 'hour']

#     # Define the target column
# target_column = 'Result'

# # set a fixed random seed for reporducibility
# random.seed(42)
# np.random.seed(42)

# # Let's split our train, validation and test datasets

# #Assuming you have a column named "Season" in your DataFrame
# train_data = df[df['Season'] <= 2022]
# test_data = df[df['Season'] > 2022]

# #Further split the training data into training and validation sets
# train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# #Separate features (X) and target variable (y) for training, validation, and test sets
# X_train = train_data.drop("Result", axis=1)  # Assuming "Result" is the target variable
# y_train = train_data["Result"]

# X_val = val_data.drop("Result", axis=1)
# y_val = val_data["Result"]

# X_test = test_data.drop("Result", axis=1)
# y_test = test_data["Result"]

# # # Verify the shapes of the sets
# # print("Train set shape:", X_train.shape, y_train.shape)
# # print("Validation set shape:", X_val.shape, y_val.shape)
# # print("Test set shape:", X_test.shape, y_test.shape)

#     # Create a column transformer for numeric variables
# preprocessor_numeric = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_columns),
#         ])


#     # Create a PCA transformer for numeric variables.
#     # Specify the number of principal components
# pca = PCA(n_components=5)

#     # PCA does not distinguish between real and synthetic samples.
# imbalanced_pipeline = imbalanced_Pipeline([
#     ('preprocessor_numeric', preprocessor_numeric),
#      ('pca', pca),
#       ('smote', SMOTE()),  # You might want to adjust SMOTE parameters
# ])
# # Applying the pipeline

# X_train_resampled, y_train_resampled = imbalanced_pipeline.fit_resample(X_train, y_train)
# X_val_resampled, y_val_resampled,  = imbalanced_pipeline.fit_resample(X_val, y_val)
# X_test_resampled, y_test_resampled = imbalanced_pipeline.fit_resample(X_test, y_test)
# # Sidebar for user input
# st.sidebar.header('Match Details')
# home_team = st.sidebar.selectbox('Select Home Team', df['team_encoded'].unique())
# away_team = st.sidebar.selectbox('Select Away Team', df['team_encoded'].unique())

# # Filter the data for the selected teams
# selected_data = df[
#     ((df['team_encoded'] == home_team) & (df['opp_encoded'] == away_team)) |
#     ((df['opp_encoded'] == home_team) & (df['team_encoded'] == away_team))
# ]

# if not selected_data.empty:
#     # Prepare the input features for prediction
#     features = selected_data.drop(columns=['Result'])

#     # Make the prediction
#     prediction = model.predict(features)

#     st.subheader('Match Prediction')
#     st.write(f'Predicted Outcome: {prediction[0]}')
# else:
#     st.warning('No data available for the selected teams.')


# In[33]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbalanced_Pipeline
from sklearn.model_selection import train_test_split
import random
import numpy as np

 # for model persistence

# Load your pre-trained Gradient Boosting model
model = joblib.load('gb_classifier.pkl')
# preprocess_function = joblib.load("preprocess_data_function.pkl")
# pipeline_function = joblib.load("pipeline_and_function.pkl")

# Load the CSV file
df = pd.read_csv('Latest_epl_data.csv')

# Apply preprocessing to the loaded data
    # Convert 'Date' and 'Time_x' to datetime
df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time_x'])
df['Time'] = pd.to_datetime(df['Time_x'], format='%H:%M').dt.time

    # Fill missing values in 'Attendance' with seasonal median
df['Season'] = df['Date'].dt.year
seasonal_median = df.groupby('Season')['Attendance'].transform('median')
df['Attendance'].fillna(seasonal_median, inplace=True)

    # Fill missing values in 'Dist' with overall median
median_dist = df['Dist'].median()
df['Dist'].fillna(median_dist, inplace=True)

    # Drop unnecessary columns
columns_to_drop = ['Time_x', 'Comp', 'xGA', 'xG', 'Captain', 'Referee', 'Match Report', 'Notes', 'Cmp.1', 'Cmp.2', 'Cmp.3', 'Att.1', 'Att.2',
                       'Att.3', 'Cmp%', 'Cmp%.1', 'Cmp%.2', 'Cmp%.3', 'Cmp.4', 'Cmp.5', 'Cmp.6', 'Cmp.7', 'Time_y']
df.drop(columns=columns_to_drop, axis=1, inplace=True)
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion ": "Brighton", "Manchester United ": "Manchester Utd", "Newcastle United ": "Newcastle Utd", "Tottenham Hotspur ": "Tottenham", "West Ham United ": "West Ham",
              "Wolverhampton Wanderers ": "Wolves",'Newcastle United ': 'Newcastle Utd', 'West Ham United ': 'West Ham', 'Sheffield United ' : 'Sheffield Utd' }
mapping = MissingDict(**map_values)
df["Team"] = df["Team"].map(mapping)

    # Map 'W', 'D', 'L' to numeric values in the 'Result' column
result_mapping = {'W': 1, 'D': 0, 'L': 0}
df['Result'] = df['Result'].map(result_mapping)

    # Feature engineering
df['Cmp_Poss_Ratio'] = df['Cmp'] / df['Poss']
df['Cmp_PrgDist_Ratio'] = df['Cmp'] / df['PrgDist']

    # Drop specified features
features_to_drop = ['TotDist', 'Att', 'Poss', 'PrgDist']
df.drop(features_to_drop, axis=1, inplace=True)

    # Encode categorical features
df["Venue"] = df["Venue"].astype("category").cat.codes
df["opp_encoded"] = df["Opponent"].astype("category").cat.codes
df["team_encoded"] = df["Team"].astype("category").cat.codes

    # Convert 'Time' to whole number
df['hour'] = df['Time'].astype(str).str.extract(r'(\d+)').astype(int)

    # Map days of the week to code
df["day_encoded"] = df["Date"].dt.dayofweek

    # Convert 'Result' column from strings to integers
df["Target"] = (df["Result"] == "W").astype("int")

    # Encode 'Formation' column
df["formation_encoded"] = df["Formation"].astype("category").cat.codes

    # Create seasons
df['Date'] = pd.to_datetime(df['Date'])

    # Define a function to determine the season
def get_season(date):
    if date.month >= 8:
        return date.year + 1
    else:
        return date.year

    # Apply the function to create a new 'Season' column
df['Season'] = df['Date'].apply(get_season)

    # Convert 'Round' column to integer codes using label encoding
df['Round_encoded'] = df['Round'].astype('category').cat.codes

    # Drop encoded categorical features
df.drop(columns=['Round', 'Day','Formation', 'Time', 'Date'], inplace=True)
    # Define the numerical columns (excluding 'Formation')
numerical_columns = ['GF', 'GA', 'Attendance', 'Sh', 'SoT', 'FK',
                     'Dist', 'PK', 'PKatt', 'Cmp', 'Cmp_Poss_Ratio', 'Cmp_PrgDist_Ratio', 'hour']

    # Define the target column
target_column = 'Result'

# Create a column transformer for numeric variables
preprocessor_numeric = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
    ])

# Create a PCA transformer for numeric variables.
# Specify the number of principal components
pca = PCA(n_components=5)

# PCA does not distinguish between real and synthetic samples.
imbalanced_pipeline = imbalanced_Pipeline([
    ('preprocessor_numeric', preprocessor_numeric),
    ('pca', pca),
    ('smote', SMOTE()),  # You might want to adjust SMOTE parameters
])

# Set a fixed random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Assuming you have a column named "Season" in your DataFrame
train_data = df[df['Season'] <= 2022]
test_data = df[df['Season'] > 2022]

# Further split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Separate features (X) and target variable (y) for training, validation, and test sets
X_train = train_data.drop("Result", axis=1)
y_train = train_data["Result"]

X_val = val_data.drop("Result", axis=1)
y_val = val_data["Result"]

X_test = test_data.drop("Result", axis=1)
y_test = test_data["Result"]

# Applying the pipeline to training, validation, and test sets
X_train_resampled, y_train_resampled = imbalanced_pipeline.fit_resample(X_train, y_train)
X_val_resampled, y_val_resampled = imbalanced_pipeline.fit_resample(X_val, y_val)
X_test_resampled, y_test_resampled = imbalanced_pipeline.fit_resample(X_test, y_test)

# Set the wallpaper URL
wallpaper_url = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.fifplay.com%2Fpremier-league-2023-2024-predictions%2F&psig=AOvVaw2TD-wPijDxF3xqT6Pc8TKM&ust=1697573560984000&source=images&cd=vfe&opi=89978449&ved=0CA8QjRxqFwoTCLiGtsqw-4EDFQAAAAAdAAAAABAD.jpg"  # Replace with the actual URL of your wallpaper

# Add the wallpaper and styling to the background
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{wallpaper_url}');
            background-size: cover;
        }}

        .sidebar {{
            background-color: #3498db;  /* Blue color for the sidebar */
            padding: 10px;
            border-radius: 10px;
            margin: 10px;
        }}

        .predict-button {{
            background-color: #ffffff;  /* White color for the predict button */
            color: #3498db;  /* Blue color for the text on the button */
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            cursor: pointer;
        }}

        .predict-button:hover {{
            background-color: #3498db;  /* Change color on hover */
            color: #ffffff;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input with custom styling
st.sidebar.header('Match Details')
home_team = st.sidebar.selectbox('Select Home Team', df['Team'].unique())
away_team = st.sidebar.selectbox('Select Away Team', df['Team'].unique())

# Add a predict button with custom styling
predict_button = st.sidebar.button('Predict Match', key='predict-button')

if predict_button:
    # Filter the data for the selected teams
    selected_data = df[((df['Team'] == home_team) & (df['Opponent'] == away_team)) | ((df['Opponent'] == home_team) | (df['Team'] == away_team))]

    if not selected_data.empty:
        # Apply the same preprocessing to the selected data
        selected_features, selected_labels = imbalanced_pipeline.fit_resample(selected_data.drop(columns=['Result']), 
                                                                              selected_data['Result'])

        # Make the prediction
        prediction = model.predict(selected_features)

        # Determine the outcome based on the prediction (0 for loss or draw, 1 for win)
        outcome = "Win" if prediction[0] == 1 else "Loss or Draw"
        
        # Display the result
        st.subheader('Match Prediction')
        st.write(f'{home_team} vs. {away_team}: {home_team} {outcome}')
    else:
        st.warning('No data available for the selected teams.')

