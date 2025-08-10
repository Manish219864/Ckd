# # train.py
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from joblib import dump

# # Load dataset
# df = pd.read_csv('kidney_disease.csv')

# # Drop irrelevant columns
# df = df.drop(columns=['id'], errors='ignore')

# # Strip whitespace and handle stray characters
# for col in df.select_dtypes(include='object').columns:
#     df[col] = df[col].str.strip()

# # Encode target: 'ckd'->1, 'notckd'->0
# df['classification'] = df['classification'].map({'ckd':1, 'notckd':0})

# # Convert numeric-like strings to floats (pcv, wc, rc)
# for col in ['pcv','wc','rc']:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Define mappings for binary/categorical fields
# maps = {
#     'rbc': {'normal':0, 'abnormal':1},
#     'pc':  {'normal':0, 'abnormal':1},
#     'pcc': {'notpresent':0, 'present':1},
#     'ba':  {'notpresent':0, 'present':1},
#     'htn': {'no':0, 'yes':1},
#     'dm':  {'no':0, 'yes':1},
#     'cad': {'no':0, 'yes':1},
#     'appet': {'poor':0, 'good':1},
#     'pe':  {'no':0, 'yes':1},
#     'ane': {'no':0, 'yes':1}
# }
# for col, mapping in maps.items():
#     if col in df.columns:
#         df[col] = df[col].map(mapping)

# # List of numeric feature columns
# num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']

# # Impute missing numeric values with median
# for col in num_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#     median = df[col].median()
#     df[col].fillna(median, inplace=True)

# # Impute missing categorical/binary with mode
# for col in maps.keys():
#     if col in df.columns:
#         mode_val = df[col].mode()[0]
#         df[col].fillna(mode_val, inplace=True)

# # Separate features and target
# X = df.drop(columns=['classification'])
# y = df['classification']

# # Split data and train logistic regression with scaling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# pipe = Pipeline([
#     ('scaler', StandardScaler()), 
#     ('lr', LogisticRegression(max_iter=1000))
# ])
# pipe.fit(X_train, y_train)

# # Save the trained pipeline to disk
# dump(pipe, 'model.pkl')


#change 1

# train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv('kidney_disease.csv')

# Drop irrelevant columns
df = df.drop(columns=['id'], errors='ignore')

# Strip whitespace from strings
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Encode target: 'ckd' -> 1, 'notckd' -> 0
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# Convert numeric-like strings to floats
for col in ['pcv', 'wc', 'rc']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define mappings for categorical fields
maps = {
    'rbc': {'normal': 0, 'abnormal': 1},
    'pc':  {'normal': 0, 'abnormal': 1},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba':  {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm':  {'no': 0, 'yes': 1},
    'cad': {'no': 0, 'yes': 1},
    'appet': {'poor': 0, 'good': 1},
    'pe':  {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1}
}
for col, mapping in maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Numeric columns
num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

# Impute numeric with median
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median = df[col].median()
    df[col].fillna(median, inplace=True)

# Impute categorical with mode
for col in maps.keys():
    if col in df.columns:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

# Final feature order (lock it for both training and prediction)
feature_order = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# Ensure all features exist in dataset
X = df[feature_order]
y = df['classification']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: scaling + logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)

# Save model and feature order together
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': pipe, 'features': feature_order}, f, protocol=4)

print("✅ Model trained and saved with fixed feature order.")
