from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model and scaler
model = joblib.load('SavedModels/logistic_regression_model.pkl')
scaler = joblib.load('SavedModels/scaler.pkl')


def login_view(request):
    return render(request, 'login.html')

def homepage(request):
    return render(request,'index.html')
def diabities_form(request):
    return render(request,'diabitiesform.html')

def diabetes_result(request):
    if request.method == 'GET':
        try:
            # Retrieve form inputs
            age = float(request.GET.get('age'))
            pregnancies = float(request.GET.get('pregnancy'))
            bp = float(request.GET.get('bp'))
            glucose = float(request.GET.get('Glucose'))
            skin_thickness = float(request.GET.get('SkinThickness'))
            insulin = float(request.GET.get('Insulin'))
            bmi = float(request.GET.get('BMI'))
            diabetes_pedigree_function = float(request.GET.get('DiabetesPedigreeFunction'))

            # Preprocess input data
            input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)

            # Determine result based on prediction
            result = 'High Risk of Diabetes' if prediction[0] == 1 else 'Low Risk of Diabetes'

            return render(request, 'diabetesresult.html', {'result': result})

        except Exception as e:
            # Handle any errors gracefully (e.g., invalid input format)
            error_message = f"Error occurred: {str(e)}"
            return render(request, 'diabetesresult.html', {'result': error_message})

    # Handle invalid request methods (e.g., POST requests)
    return render(request, 'diabetesresult.html', {'result': 'Invalid Request'})