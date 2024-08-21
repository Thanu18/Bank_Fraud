from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model, encoder, and feature names
with open('xgb_model_1.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

with open('encoder_1.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

with open('feature_names_1.pkl', 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'trans_date_trans_time': int(request.form['trans_date_trans_time']),
            'dob': int(request.form['dob']),
            'hour_transaction': request.form['hour_transaction'],
            'category': request.form['category'],
            'city': request.form['city'],
            'state': request.form['state'],
            'amount': float(request.form['amount'])  # New feature
        }

        # Create a DataFrame from the form data
        df = pd.DataFrame([data])

        # Feature Engineering
        df['age'] = 2024 - df['dob']
        df['hour_transaction'] = df['trans_date_trans_time'].apply(
            lambda x: 'Morning' if 4 < x <= 12 else ('Afternoon' if 12 < x <= 20 else 'Night')
        )

        # Drop 'dob' as it's not used in the final features
        df.drop('dob', axis=1, inplace=True)

        # Encode categorical features
        encoded_data = encoder.transform(df[['hour_transaction', 'category', 'city', 'state']])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        
        # Combine original features with encoded features
        df = pd.concat([df.drop(['hour_transaction', 'category', 'city', 'state'], axis=1).reset_index(drop=True),
                        encoded_df.reset_index(drop=True)], axis=1)

        # Align columns to match the model's training data
        df = df.reindex(columns=feature_names, fill_value=0)

        # Make a prediction
        prediction = xgb_model.predict(df)

        # Return result to user
        result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
        return render_template('result.html', result=result)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
