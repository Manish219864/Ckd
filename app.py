# # app.py
# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# from joblib import load

# app = Flask(__name__)

# # Load the trained model pipeline
# model = load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract form data and convert to DataFrame
#         form_data = request.form.to_dict()
#         # Convert values to numeric (float or int)
#         data = pd.DataFrame([form_data], columns=form_data.keys()).astype(float)
#         # Predict and map to label
#         pred = model.predict(data)[0]
#         result = "CKD" if pred == 1 else "No CKD"
#         return render_template('result.html', prediction=result)
#     except Exception as e:
#         return render_template('result.html', prediction=f"Error: {e}")

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         # Parse JSON payload
#         data_json = request.get_json()
#         data = pd.DataFrame([data_json], columns=data_json.keys()).astype(float)
#         pred = model.predict(data)[0]
#         label = int(pred)
#         return jsonify({'prediction': label, 
#                         'message': ("CKD" if label==1 else "No CKD")})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)

#change 1
# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# from joblib import load

# app = Flask(__name__)

# # Load the trained model pipeline
# model = load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('home.html')

# # Allow both GET and POST on /predict
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             # Extract form data and convert to DataFrame
#             form_data = request.form.to_dict()
#             # Convert values to numeric (float or int)
#             data = pd.DataFrame([form_data], columns=form_data.keys()).astype(float)
#             # Predict and map to label
#             pred = model.predict(data)[0]
#             result = "CKD" if pred == 1 else "No CKD"
#             return render_template('result.html', prediction=result)
#         except Exception as e:
#             return render_template('result.html', prediction=f"Error: {e}")
#     else:
#         # For GET request, show the form to input data
#         return render_template('form.html')

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         # Parse JSON payload
#         data_json = request.get_json()
#         data = pd.DataFrame([data_json], columns=data_json.keys()).astype(float)
#         pred = model.predict(data)[0]
#         label = int(pred)
#         return jsonify({'prediction': label, 
#                         'message': ("CKD" if label==1 else "No CKD")})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


#change 2

# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# from joblib import load

# app = Flask(__name__)

# # Load the trained model pipeline
# model = load('model.pkl')

# # Mapping dictionaries (adjust to match your training encoding)
# categorical_mappings = {
#     "rbc": {"normal": 1, "abnormal": 0},
#     "pc": {"normal": 1, "abnormal": 0},
#     "pcc": {"present": 1, "notpresent": 0},
#     "ba": {"present": 1, "notpresent": 0},
#     "htn": {"yes": 1, "no": 0},
#     "dm": {"yes": 1, "no": 0},
#     "cad": {"yes": 1, "no": 0},
#     "appet": {"good": 1, "poor": 0},
#     "pe": {"yes": 1, "no": 0},
#     "ane": {"yes": 1, "no": 0}
# }

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             form_data = request.form.to_dict()

#             # Apply mappings for categorical fields
#             for col, mapping in categorical_mappings.items():
#                 if col in form_data:
#                     form_data[col] = mapping[form_data[col].lower()]

#             # Convert all to numeric DataFrame
#             data = pd.DataFrame([form_data], columns=form_data.keys()).astype(float)

#             # Predict
#             pred = model.predict(data)[0]
#             result = "CKD" if pred == 1 else "No CKD"
#             return render_template('result.html', prediction=result)

#         except Exception as e:
#             return render_template('result.html', prediction=f"Error: {e}")
#     else:
#         return render_template('form.html')

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         data_json = request.get_json()

#         # Apply mappings for categorical fields
#         for col, mapping in categorical_mappings.items():
#             if col in data_json:
#                 data_json[col] = mapping[data_json[col].lower()]

#         data = pd.DataFrame([data_json], columns=data_json.keys()).astype(float)

#         pred = model.predict(data)[0]
#         label = int(pred)
#         return jsonify({'prediction': label, 'message': ("CKD" if label == 1 else "No CKD")})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


#change 3

# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# from joblib import load

# app = Flask(__name__)

# # Load trained model pipeline
# model = load('model.pkl')

# # Exact features used during training
# feature_names = [
#     'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
#     'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
#     'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
#     'appet', 'pe', 'ane'
# ]

# # Mapping from train.py (must match training exactly)
# categorical_mappings = {
#     'rbc': {'normal': 0, 'abnormal': 1},
#     'pc': {'normal': 0, 'abnormal': 1},
#     'pcc': {'notpresent': 0, 'present': 1},
#     'ba': {'notpresent': 0, 'present': 1},
#     'htn': {'no': 0, 'yes': 1},
#     'dm': {'no': 0, 'yes': 1},
#     'cad': {'no': 0, 'yes': 1},
#     'appet': {'poor': 0, 'good': 1},
#     'pe': {'no': 0, 'yes': 1},
#     'ane': {'no': 0, 'yes': 1}
# }

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             form_data = request.form.to_dict()

#             # Apply mappings for categorical fields
#             for col, mapping in categorical_mappings.items():
#                 if col in form_data:
#                     form_data[col] = mapping[form_data[col].strip().lower()]

#             # Ensure all features are present, fill missing with 0
#             for col in feature_names:
#                 if col not in form_data:
#                     form_data[col] = 0

#             # Create DataFrame with correct column order
#             data = pd.DataFrame([[form_data[col] for col in feature_names]], columns=feature_names).astype(float)

#             # Predict
#             pred = model.predict(data)[0]
#             result = "CKD" if pred == 1 else "No CKD"
#             return render_template('result.html', prediction=result)

#         except Exception as e:
#             return render_template('result.html', prediction=f"Error: {e}")
#     else:
#         return render_template('form.html')

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         data_json = request.get_json()

#         # Apply mappings for categorical fields
#         for col, mapping in categorical_mappings.items():
#             if col in data_json:
#                 data_json[col] = mapping[data_json[col].strip().lower()]

#         # Ensure all features are present, fill missing with 0
#         for col in feature_names:
#             if col not in data_json:
#                 data_json[col] = 0

#         # Create DataFrame with correct column order
#         data = pd.DataFrame([[data_json[col] for col in feature_names]], columns=feature_names).astype(float)

#         pred = model.predict(data)[0]
#         label = int(pred)
#         return jsonify({'prediction': label, 'message': ("CKD" if label == 1 else "No CKD")})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


#change 4
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# from joblib import load

# app = Flask(__name__)

# # Load trained model pipeline
# model = load('model.pkl')

# # Exact features used during training
# feature_names = [
#     'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
#     'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
#     'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
#     'appet', 'pe', 'ane'
# ]

# # Mapping from train.py (must match training exactly)
# categorical_mappings = {
#     'rbc': {'normal': 0, 'abnormal': 1},
#     'pc': {'normal': 0, 'abnormal': 1},
#     'pcc': {'notpresent': 0, 'present': 1},
#     'ba': {'notpresent': 0, 'present': 1},
#     'htn': {'no': 0, 'yes': 1},
#     'dm': {'no': 0, 'yes': 1},
#     'cad': {'no': 0, 'yes': 1},
#     'appet': {'poor': 0, 'good': 1},
#     'pe': {'no': 0, 'yes': 1},
#     'ane': {'no': 0, 'yes': 1}
# }

# # Default values if user leaves fields empty
# numeric_defaults = {col: 0 for col in feature_names if col not in categorical_mappings}
# categorical_defaults = {col: list(mapping.values())[0] for col, mapping in categorical_mappings.items()}

# def preprocess_input(input_dict):
#     """Convert incoming data into a clean DataFrame matching training features."""
#     clean_data = {}

#     for col in feature_names:
#         val = input_dict.get(col, None)

#         # Handle missing values
#         if val is None or val == '' or str(val).strip().lower() == 'nan':
#             if col in categorical_mappings:
#                 clean_data[col] = categorical_defaults[col]
#             else:
#                 clean_data[col] = numeric_defaults[col]
#             continue

#         # Apply categorical mapping if needed
#         if col in categorical_mappings:
#             val_str = str(val).strip().lower()
#             mapped = categorical_mappings[col].get(val_str, categorical_defaults[col])
#             clean_data[col] = mapped
#         else:
#             try:
#                 clean_data[col] = float(val)
#             except ValueError:
#                 clean_data[col] = numeric_defaults[col]

#     return pd.DataFrame([[clean_data[col] for col in feature_names]], columns=feature_names)

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             form_data = request.form.to_dict()
#             data = preprocess_input(form_data)

#             # Predict
#             pred = model.predict(data)[0]
#             result = "CKD" if pred == 1 else "No CKD"
#             return render_template('result.html', prediction=result)
#         except Exception as e:
#             return render_template('result.html', prediction=f"Error: {e}")
#     else:
#         return render_template('form.html')

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         data_json = request.get_json()
#         data = preprocess_input(data_json)

#         pred = model.predict(data)[0]
#         label = int(pred)
#         return jsonify({'prediction': label, 'message': ("CKD" if label == 1 else "No CKD")})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)

#change 5

# import pickle
# import pandas as pd
# import numpy as np
# from flask import Flask, request, render_template

# app = Flask(__name__)

# # Load model
# model = pickle.load(open("model.pkl", "rb"))

# # Exact feature names used during training
# FEATURE_NAMES = [
#     'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
#     'bu', 'sc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
# ]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect form data
#         form_data = {feature: request.form.get(feature) for feature in FEATURE_NAMES}

#         # Convert to DataFrame
#         df = pd.DataFrame([form_data])

#         # Convert to numeric where possible
#         df = df.apply(pd.to_numeric, errors='ignore')

#         # Replace missing values with NaN
#         df = df.replace("", np.nan)

#         # Impute NaN with median for numeric columns and mode for categorical
#         for col in df.columns:
#             if df[col].dtype.kind in 'biufc':  # numeric
#                 df[col].fillna(df[col].median(), inplace=True)
#             else:  # categorical
#                 df[col].fillna(df[col].mode()[0], inplace=True)

#         # Ensure correct order
#         df = df[FEATURE_NAMES]

#         # Predict
#         prediction = model.predict(df)[0]

#         return render_template('result.html', prediction=prediction)

#     except Exception as e:
#         return render_template('result.html', prediction=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True)

#change 6 (redi to 4)
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# from joblib import load

# app = Flask(__name__)

# # Load trained model pipeline
# model = load('model.pkl')

# # Exact features used during training
# feature_names = [
#     'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
#     'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
#     'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
#     'appet', 'pe', 'ane'
# ]

# # Mapping from train.py (must match training exactly)
# categorical_mappings = {
#     'rbc': {'normal': 0, 'abnormal': 1},
#     'pc': {'normal': 0, 'abnormal': 1},
#     'pcc': {'notpresent': 0, 'present': 1},
#     'ba': {'notpresent': 0, 'present': 1},
#     'htn': {'no': 0, 'yes': 1},
#     'dm': {'no': 0, 'yes': 1},
#     'cad': {'no': 0, 'yes': 1},
#     'appet': {'poor': 0, 'good': 1},
#     'pe': {'no': 0, 'yes': 1},
#     'ane': {'no': 0, 'yes': 1}
# }

# def preprocess_input(input_dict):
#     """
#     Converts input dictionary into a clean DataFrame for prediction.
#     Handles:
#     - Categorical mapping
#     - Missing values
#     - Data type conversion
#     """
#     clean_data = {}

#     for col in feature_names:
#         if col in input_dict and input_dict[col] not in [None, '', 'nan']:
#             val = input_dict[col]
#             if col in categorical_mappings:
#                 # Map categorical safely
#                 val = str(val).strip().lower()
#                 if val in categorical_mappings[col]:
#                     clean_data[col] = categorical_mappings[col][val]
#                 else:
#                     clean_data[col] = np.nan
#             else:
#                 # Numeric conversion
#                 try:
#                     clean_data[col] = float(val)
#                 except:
#                     clean_data[col] = np.nan
#         else:
#             clean_data[col] = np.nan

#     # Convert to DataFrame
#     df = pd.DataFrame([clean_data])

#     # Handle missing values
#     for col in df.columns:
#         if df[col].isna().any():
#             if col in categorical_mappings:
#                 df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
#             else:
#                 df[col].fillna(df[col].median() if not df[col].median() == np.nan else 0, inplace=True)

#     return df.astype(float)


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             form_data = request.form.to_dict()
#             data = preprocess_input(form_data)

#             pred = model.predict(data)[0]
#             result = "CKD" if pred == 1 else "No CKD"
#             return render_template('result.html', prediction=result)

#         except Exception as e:
#             return render_template('result.html', prediction=f"Error: {e}")
#     else:
#         return render_template('form.html')

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         data_json = request.get_json()
#         data = preprocess_input(data_json)

#         pred = model.predict(data)[0]
#         label = int(pred)
#         return jsonify({'prediction': label, 'message': ("CKD" if label == 1 else "No CKD")})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


#change 7

# from flask import Flask, render_template, request
# import pickle
# import pandas as pd

# app = Flask(__name__)

# # Load the trained model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# # IMPORTANT: This must match the exact column order used during training
# feature_order = [
#     'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
#     'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
#     'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
# ]

# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         # Get form data
#         form_values = {}
#         for feature in feature_order:
#             form_values[feature] = request.form.get(feature)

#         # Ensure numeric conversion for numerical fields
#         numeric_fields = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
#         for field in numeric_fields:
#             form_values[field] = float(form_values[field])

#         # Create DataFrame with correct column order
#         input_df = pd.DataFrame([[form_values[col] for col in feature_order]], columns=feature_order)

#         # Predict
#         prediction = model.predict(input_df)[0]
#         result_text = "The patient is likely to have CKD." if prediction == 1 else "The patient is unlikely to have CKD."

#         return render_template("result.html", prediction=result_text)

#     return render_template("form.html")

# if __name__ == "__main__":
#     app.run(debug=True)

#change 8

# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # ----------------------------
# # Load model and feature order
# # ----------------------------
# with open("model.pkl", "rb") as f:
#     data = pickle.load(f)
#     model = data['model']
#     feature_order = data['features']

# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get JSON data from request
#         input_data = request.get_json()

#         # Ensure all expected features are present
#         values = []
#         for feature in feature_order:
#             if feature not in input_data:
#                 return jsonify({"error": f"Missing feature: {feature}"}), 400
#             values.append(input_data[feature])

#         # Convert to DataFrame for model input
#         input_df = pd.DataFrame([values], columns=feature_order)

#         # Predict
#         prediction = model.predict(input_df)[0]
#         probability = model.predict_proba(input_df)[0][1]

#         return jsonify({
#             "prediction": int(prediction),
#             "probability": float(probability)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


#change 9

# from flask import Flask, request, render_template
# import pickle
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load trained model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Home route
# @app.route("/")
# def home():
#     return render_template("form.html")  # Change to your HTML file name

# # Predict route
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         try:
#             # Get data from form
#             features = [float(x) for x in request.form.values()]

#             # Ensure features match training order
#             input_df = pd.DataFrame([features], columns=[
#                 "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
#                 "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn",
#                 "dm", "cad", "appet", "pe", "ane"
#             ])

#             # Make prediction
#             prediction = model.predict(input_df)[0]
#             result = "Chronic Kidney Disease" if prediction == 1 else "No Chronic Kidney Disease"

#             return render_template("result.html", prediction=result)

#         except Exception as e:
#             return f"Error: {str(e)}"

#     # GET request → load form
#     return render_template("form.html")  # Same form page

# if __name__ == "__main__":
#     app.run(debug=True)


#change 10

# from flask import Flask, request, render_template
# import pickle
# import pandas as pd

# app = Flask(__name__)

# # Load trained model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Define categorical mappings (must match training preprocessing)
# categorical_mappings = {
#     "rbc": {"normal": 0, "abnormal": 1},
#     "pc": {"normal": 0, "abnormal": 1},
#     "pcc": {"notpresent": 0, "present": 1},
#     "ba": {"notpresent": 0, "present": 1},
#     "htn": {"no": 0, "yes": 1},
#     "dm": {"no": 0, "yes": 1},
#     "cad": {"no": 0, "yes": 1},
#     "appet": {"good": 0, "poor": 1},
#     "pe": {"no": 0, "yes": 1},
#     "ane": {"no": 0, "yes": 1}
# }

# # Feature order from training
# feature_order = [
#     "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
#     "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn",
#     "dm", "cad", "appet", "pe", "ane"
# ]

# @app.route("/")
# def home():
#     return render_template("form.html")

# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         try:
#             form_data = request.form.to_dict()

#             processed_features = []
#             for feature in feature_order:
#                 value = form_data.get(feature)

#                 # Map categorical values
#                 if feature in categorical_mappings:
#                     value = categorical_mappings[feature].get(value.lower(), value)

#                 # Convert to float
#                 processed_features.append(float(value))

#             # Create DataFrame in the correct order
#             input_df = pd.DataFrame([processed_features], columns=feature_order)

#             prediction = model.predict(input_df)[0]
#             result = "Chronic Kidney Disease" if prediction == 1 else "No Chronic Kidney Disease"

#             return render_template("result.html", prediction=result)

#         except Exception as e:
#             return f"Error: {str(e)}"

#     return render_template("form.html")

# if __name__ == "__main__":
#     app.run(debug=True)

#change 11

from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model (changed just now)
with open("model.pkl", "rb") as f:
    loaded_obj = pickle.load(f)
    model = loaded_obj["model"] if isinstance(loaded_obj, dict) else loaded_obj


# Feature order from training
feature_order = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
    "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

# Same mappings used in train.py
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

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_values = {}

        for feature in feature_order:
            raw_val = request.form.get(feature)

            if feature in maps:  # categorical
                if raw_val is None:
                    raw_val = list(maps[feature].values())[0]  # default to first mapping
                else:
                    raw_val = maps[feature].get(raw_val.lower(), list(maps[feature].values())[0])
                form_values[feature] = raw_val
            else:  # numeric
                if raw_val is None or raw_val.strip() == "":
                    raw_val = 0  # default numeric
                form_values[feature] = float(raw_val)

        # Create DataFrame
        input_df = pd.DataFrame([[form_values[col] for col in feature_order]], columns=feature_order)

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Chronic Kidney Disease" if prediction == 1 else "No Chronic Kidney Disease"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
