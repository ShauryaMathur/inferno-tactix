from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from create_env import generate_heightmap
from generate_timeseries import get_75day_timeseries
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize app
app = Flask(__name__)
CORS(app)

SEQ_LEN = 75
# Load the trained model
class CNN_LSTM_Wildfire(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0,2,1)  # Change shape for CNN input
        x = self.cnn(x)
        x = x.permute(0,2,1)  # Change back shape for LSTM input
        _, (hn, _) = self.lstm(x)
        out = hn[-1]
        return self.fc(out).squeeze(1)

# Instantiate the model and load the saved weights
model = CNN_LSTM_Wildfire(
    input_features=14,  # Assuming you have 14 features in your dataset
    hidden_size=128,
    num_layers=3,
    dropout=0.25
)
model.load_state_dict(torch.load('infernix_model.pth', map_location='mps'))
model.eval()  # Set to evaluation mode
# Load models
# MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
# model1 = torch.load(os.path.join(MODEL_DIR, 'model1.pth'), map_location='cpu')
# model1.eval()

# model2 = torch.load(os.path.join(MODEL_DIR, 'model2.pth'), map_location='cpu')
# model2.eval()

# Simple health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/api/createEnvironment', methods=['POST'])
def create_environment():
    # Pull them off request.args (they’ll be strings)
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if lat is None or lon is None:
        return jsonify({
            'error': 'Missing required query parameters lat and lon'
        }), 400

    # Now pass them into your function
    # (assuming your generate_heightmap takes (lat_str, lon_str))
    try:
        param = str(lon + ',' + lat)
        heightmap = generate_heightmap(param)
    except Exception as e:
        return jsonify({
            'error': 'Heightmap generation failed',
            'details': str(e)
        }), 500

    # Return whatever generate_heightmap gives you,
    # or wrap it in JSON as appropriate:
    return jsonify({
        'lat': lat,
        'lon': lon,
        'heightmap': heightmap
    })

@app.route('/api/predictWildfire', methods=['POST'])
def get_timeseries():

    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    dt_str = request.args.get('date')
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%d')

    print(dt_obj,type(dt_obj))
    out_df = get_75day_timeseries(lat, lon, dt_obj)

    print(out_df.info())

    # Check if the dataframe has valid data
    if out_df is None or out_df.empty:
        return jsonify({'error': 'Invalid data for the given coordinates and date'}), 400

    # Preprocess the data (e.g., scaling, reshaping)
    features = ['pr','rmax','rmin','sph','srad','tmmn','tmmx','vs','bi','fm100','fm1000','erc','pet','vpd']
    data = out_df[features].values
    scaler = StandardScaler()  # Make sure to fit this scaler during training, if you used scaling
    data_scaled = scaler.fit_transform(data)  # Apply scaling

    # Reshape data to match the expected input shape of the model (batch_size, seq_len, features)
    input_data = np.reshape(data, (1, SEQ_LEN, len(features)))  # Assuming seq_len=75
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Make the prediction
    with torch.no_grad():  # Disable gradient computation for inference
        try:
            logits = model(input_tensor)  # Get the raw output from the model
            print(logits)
            prediction = torch.sigmoid(logits).item()  # Apply sigmoid and convert to scalar
        except Exception as e:
            print(e)


    print(f"Prediction: {prediction}")

    return jsonify({
        'lat': lat,
        'lon': lon,
        'date': dt_obj.strftime('%Y-%m-%d'),
        'prediction': prediction
    })



# Endpoint 1: inference with model1
# @app.route('/api/predict1', methods=['POST'])
# def predict1():
#     data = request.get_json()
#     # assume data['input'] is a list or tensor-like
#     tensor = torch.tensor(data['input'], dtype=torch.float32)
#     with torch.no_grad():
#         output = model1(tensor)
#     # convert to list for JSON
#     return jsonify({'prediction': output.tolist()})

# # Endpoint 2: inference with model2
# @app.route('/api/predict2', methods=['POST'])
# def predict2():
#     data = request.get_json()
#     tensor = torch.tensor(data['input'], dtype=torch.float32)
#     with torch.no_grad():
#         output = model2(tensor)
#     return jsonify({'prediction': output.tolist()})

# # Endpoint 3: custom business logic
# @app.route('/api/combined', methods=['POST'])
# def combined():
#     data = request.get_json()
#     t = torch.tensor(data['input'], dtype=torch.float32)
#     with torch.no_grad():
#         out1 = model1(t)
#         out2 = model2(t)
#     # example: average
#     combined = (out1 + out2) / 2.0
#     return jsonify({'combined': combined.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)