from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_date = data.get("selectedDate", "Unknown Date")

    # Generate synthetic training data dynamically
    train_rows = 1000  # Example size
    window_length = 7
    number_of_features = 6

    # Generate random training data (simulate real training data)
    train_data = np.random.rand(train_rows, number_of_features)
    train_samples = np.empty([train_rows - window_length, window_length, number_of_features], dtype=float)
    train_labels = np.empty([train_rows - window_length, number_of_features], dtype=float)

    for i in range(0, train_rows - window_length):
        train_samples[i] = train_data[i : i + window_length, :]
        train_labels[i] = train_data[i + window_length, :]

    # Re-train the model with the new data
    model.fit(train_samples, train_labels, batch_size=100, epochs=10, verbose=0)

    # Create input data for prediction
    input_data = train_data[-window_length:, :]  # Use the last window for prediction
    x_next = scaler.transform(input_data)
    y_next_pred = model.predict(np.array([x_next]))

    # Process predictions
    prediction_without_rounding = scaler.inverse_transform(y_next_pred).astype(int)[0]
    prediction_rounded_up = prediction_without_rounding + 1
    prediction_rounded_down = prediction_without_rounding - 1

    # Create a response dictionary
    response = {
        "drawing_date": selected_date,
        "prediction_without_rounding": prediction_without_rounding.tolist(),
        "prediction_rounded_up": prediction_rounded_up.tolist(),
        "prediction_rounded_down": prediction_rounded_down.tolist()
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
