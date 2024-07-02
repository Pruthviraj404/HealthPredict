from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the models and scalers
model = joblib.load('SavedModels/logistic_regression_model.pkl')
scaler = joblib.load('SavedModels/scaler.pkl')

# Load the kidney disease model and label encoders
model_path = 'SavedModels/kidney_disease_rf_model.pkl'
model_data = joblib.load(model_path)
model1 = model_data['model']
label_encoders = model_data['label_encoders']

def login_view(request):
    return render(request, 'login.html')

def homepage(request):
    return render(request, 'index.html')

def diabities_form(request):
    return render(request, 'diabitiesform.html')

def diabetes_result(request):
    if request.method == 'POST':
        try:
            # Retrieve form inputs
            age = int(request.POST.get('age', 0))
            pregnancies = int(request.POST.get('pregnancy', 0))
            bp = int(request.POST.get('bp', 0))
            glucose = float(request.POST.get('Glucose', 0.0))
            skin_thickness = int(request.POST.get('SkinThickness', 0))
            insulin = int(request.POST.get('Insulin', 0))
            bmi = float(request.POST.get('BMI', 0.0))
            diabetes_pedigree_function = float(request.POST.get('DiabetesPedigreeFunction', 0.0))

            # Preprocess input data
            input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            print(f"Inputs: {input_data}")

            # Ensure scaler is correctly used
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)

            # Determine result based on prediction
            result= 'High Risk of Diabetes' if prediction[0] == 1 else 'Low Risk of Diabetes'

            return render(request, 'diabetesresult.html', {'result': result})

        except Exception as e:
            # Handle any errors gracefully (e.g., invalid input format)
            error_message = f"Error occurred: {str(e)}"
            return render(request, 'diabetesresult.html', {'result': error_message})

    # Handle invalid request methods (e.g., POST requests)
    return render(request, 'diabetesresult.html', {'result': 'Invalid Request'})

def heart_form(request):
    return render(request, 'heartform.html')

def heart_result(request):
    if request.method == 'POST':
        try:
            # Retrieve and convert form inputs to appropriate types
            age = int(request.POST.get('age'))
            sex = int(request.POST.get('sex'))
            cp = int(request.POST.get('cp'))
            trestbps = int(request.POST.get('trestbps'))
            chol = int(request.POST.get('chol'))
            fbs = int(request.POST.get('fbs'))
            restecg = int(request.POST.get('restecg'))
            thalach = int(request.POST.get('thalach'))
            exang = int(request.POST.get('exang'))
            oldpeak = float(request.POST.get('oldpeak'))
            slope = int(request.POST.get('slope'))
            ca = int(request.POST.get('ca'))
            thal = int(request.POST.get('thal'))

            # Prepare inputs
            inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            # Log the inputs for debugging
            print(f"Inputs: {inputs}")

            # Load the model
            loaded_model = joblib.load('SavedModels/decision_tree_model.pkl')

            # Prepare the input for prediction
            input_as_numpy = np.asarray(inputs)
            input_reshaped = input_as_numpy.reshape(1, -1)
            pre1 = loaded_model.predict(input_reshaped)

            # Log the prediction result for debugging
            print(f"Prediction: {pre1}")

            output = "The patient seems to have heart disease:(" if pre1[0] == 1 else "The patient seems to be Normal:)"

            return render(request, 'heartresult.html', {'result': output})
        except Exception as e:
            # Handle any errors gracefully (e.g., invalid input format)
            error_message = f"Error occurred: {str(e)}"
            return render(request, 'heartresult.html', {'result': error_message})

    return render(request, 'heartresult.html')

def kidney_form(request):
    return render(request, 'kidneydiseaseform.html')

# List of categorical columns to encode
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Preprocess user input
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    
    for column in categorical_columns:
        input_df[column] = label_encoders[column].transform(input_df[column].astype(str))
    
    input_df['pcv'] = pd.to_numeric(input_df['pcv'], errors='coerce')
    input_df['wc'] = pd.to_numeric(input_df['wc'], errors='coerce')
    input_df['rc'] = pd.to_numeric(input_df['rc'], errors='coerce')
    
    return input_df

# Predict user input
def predict_user_input(model, user_input):
    preprocessed_input = preprocess_input(user_input)
    input_array = preprocessed_input.values  # Remove feature names
    prediction = model.predict(input_array)
    prediction_label = label_encoders['classification'].inverse_transform(prediction)
    if prediction_label[0] == 'ckd':
        return "LOOKS LIKE CKD"
    else:
        return "LOOKS LIKE NO CKD"
def kidney_result(request):
    if request.method == 'POST':
        try:
            # Retrieve form inputs and convert to appropriate types
            user_input = {
                'age': int(request.POST.get('age')),
                'bp': int(request.POST.get('bp')),
                'sg': float(request.POST.get('sg')),
                'al': int(request.POST.get('al')),
                'su': int(request.POST.get('su')),
                'rbc': request.POST.get('rbc'),
                'pc': request.POST.get('pc'),
                'pcc': request.POST.get('pcc'),
                'ba': request.POST.get('ba'),
                'bgr': int(request.POST.get('bgr')),
                'bu': int(request.POST.get('bu')),
                'sc': float(request.POST.get('sc')),
                'sod': int(request.POST.get('sod')),
                'pot': float(request.POST.get('pot')),
                'hemo': float(request.POST.get('hemo')),
                'pcv': request.POST.get('pcv'),
                'wc': request.POST.get('wc'),
                'rc': request.POST.get('rc'),
                'htn': request.POST.get('htn'),
                'dm': request.POST.get('dm'),
                'cad': request.POST.get('cad'),
                'appet': request.POST.get('appet'),
                'pe': request.POST.get('pe'),
                'ane': request.POST.get('ane')
            }
            
            # Predict the user input
            prediction = predict_user_input(model1, user_input)
            # print(prediction)
            # result3 = "NOCKD" if prediction == 'NOCKD' else "CKD"
            
            # Render the result template with the prediction
            return render(request, 'kidneyresult.html', {'prediction':prediction})
        
        except Exception as e:
            return render(request, 'kidneyresult.html', {'error': str(e)})
    
    return render(request, 'form.html')
