from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained churn prediction model
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    CreditScore = int(request.form['CreditScore'])
    Geography = request.form['Geography']
    Gender = request.form['Gender']
    Age = int(request.form['Age'])
    Tenure = int(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    NumOfProducts = int(request.form['NumOfProducts'])
    HasCrCard = int(request.form['HasCrCard'])
    IsActiveMember = int(request.form['IsActiveMember'])
    EstimatedSalary = float(request.form['EstimatedSalary'])
    
    # Create a data frame from the inputs
    input_data = pd.DataFrame([[CreditScore, Geography, Gender, Age, Tenure, Balance, 
                                NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]],
                              columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                                       'EstimatedSalary'])
    
    # Convert categorical variables (Geography and Gender) to numerical format
    input_data = pd.get_dummies(input_data, columns=['Geography', 'Gender'], drop_first=True)
    
    # Ensure that all the necessary columns are present
    expected_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 
                        'Geography_Spain', 'Gender_Male']
    
    # Add missing columns if any, with default value 0
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Predict churn (Exited) using the model
    prediction = model.predict(input_data)[0]
    
    # Return the result
    result = "Customer will exit" if prediction == 1 else "Customer will not exit"
    
    return render_template('result.html', prediction_text=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
