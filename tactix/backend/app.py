from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from create_env import generate_heightmap
from generate_timeseries import get_75day_timeseries
from datetime import datetime


# Initialize app
app = Flask(__name__)
CORS(app)

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

    return jsonify({
        'lat': lat,
        'lon': lon,
        # 'df': out_df
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