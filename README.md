# Chronic Kidney Disease Prediction using Machine Learning and Flask Deployment

## Project Overview
This project aims to develop a machine learning-based web application to predict Chronic Kidney Disease (CKD) using patient medical parameters. Early detection of CKD is crucial for timely treatment, and this tool helps healthcare professionals and users assess CKD risk efficiently.

## Features
- Input form to enter patient medical details.
- Machine learning model predicts CKD presence.
- Clear, user-friendly prediction results.
- Backend built with Flask, model served via Pickle.
- Frontend built with HTML and CSS.

## Dataset
The model is trained on the [Kaggle CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease), which contains 400 records with 26 medical features and a binary target variable (`ckd` / `notckd`).

## Technologies Used
- Python 3.9+
- Pandas, NumPy, scikit-learn
- Flask (backend web framework)
- Pickle (model serialization)
- HTML, CSS (frontend templates)

## Project Workflow
1. User inputs patient medical parameters in the web form.
2. Flask backend preprocesses inputs and feeds data to the trained ML model.
3. Model predicts whether the patient has CKD.
4. Prediction result is displayed on the web page.

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Steps
```bash
# Clone the repository
git clone <repository_link>

# Navigate to the project directory
cd ckd-prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
Thank you for exploring this project!
