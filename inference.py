import mlflow.pyfunc
import pandas as pd

# Define the sample input with correct feature names
data = pd.DataFrame([{
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31
}])

model_name = "diabetes-rf"
model_version = 5

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(model.predict(data))
