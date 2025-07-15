from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from create_env import generate_data_and_heightmap
from generate_timeseries import get_75day_timeseries
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import traceback

# from io import StringIO
try:
    from io import StringIO
except ImportError:
    import StringIO
      # fallback for very old Python

import joblib
import subprocess
# from ..fire_analytics.one_pager import generate_report

# Initialize app
app = Flask(__name__)
CORS(app)

SEQ_LEN     = 75
BASE_FEATS  = ['pr','rmax','rmin','sph','srad','tmmn','tmmx',
               'vs','bi','fm100','fm1000','erc','pet','vpd']
FEAT_DIM    = len(BASE_FEATS) + 2  # +2 for sin/cos month
MODEL_PATH  = 'models/best_model.pt'
SCALER_PATH = 'models/scaler.pkl'

# ─── Model Definition ────────────────────────────────────────────────────────
class CNN_BiLSTMAttn(nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers, dropout, init_method):
        super().__init__()
        self.conv1 = nn.Conv1d(feat_dim, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,   64, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            64, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.attn_w = nn.Linear(2*hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights(init_method)

    def _init_weights(self, method):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        x = x.transpose(1, 2)                             # -> (batch, feat_dim, seq_len)
        x = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(torch.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)                             # -> (batch, seq_len, 64)
        out, _ = self.lstm(x)                             # -> (batch, seq_len, 2*hidden_dim)
        attn_scores = torch.softmax(self.attn_w(out).squeeze(-1), dim=1)
        context = torch.sum(out * attn_scores.unsqueeze(-1), dim=1)
        return self.classifier(context).squeeze(-1)       # raw logit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = joblib.load(SCALER_PATH)

# Instantiate & load model
model = CNN_BiLSTMAttn(
    feat_dim=FEAT_DIM,
    hidden_dim=150,   # must match your training params
    n_layers=3,
    dropout=0.10,
    init_method='kaiming'
).to(device)
# state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# model.load_state_dict(state_dict)
# model.eval()
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# If ckpt is a pure state_dict, this just returns ckpt itself.
# If it’s a full checkpoint, try to grab the 'model' or 'state_dict' field.
if isinstance(ckpt, dict):
    if 'model' in ckpt:
        sd = ckpt['model']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
else:
    sd = ckpt

model.load_state_dict(sd)
model.eval()

def build_input_tensor(df: pd.DataFrame, date: datetime):
    # 1) extract raw features (75×14)
    raw = df[BASE_FEATS].values

    # 2) build month sin/cos array (75×2)
    m = date.month - 1
    sin_m, cos_m = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)
    month_feats = np.tile([sin_m, cos_m], (SEQ_LEN, 1))

    # 3) concatenate to full (75×16)
    full = np.hstack([raw, month_feats])

    # 4) now scale the full 16-dim vectors
    full_scaled = scaler.transform(full)       # <-- expects 16 features

    # 5) reshape into (1,75,16) tensor
    return torch.tensor(full_scaled[np.newaxis, ...],
                        dtype=torch.float32)

# ─── Endpoint ────────────────────────────────────────────────────────────────
@app.route('/api/predictWildfire', methods=['POST'])
def predict_wildfire():
    try:
        lat   = float(request.args['lat'])
        lon   = float(request.args['lon'])
        dt_str= request.args['date']
        dt_obj= datetime.strptime(dt_str, '%Y-%m-%d')
    except Exception:
        return jsonify({'error': 'lat, lon and date (YYYY-MM-DD) are required'}), 400

    try:
        print("Fetching data...")
        out_df = get_75day_timeseries(lat, lon, dt_obj)
        print("Data fetched.", len(out_df))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error fetching data: {e}'}), 400
    
    if out_df is None or out_df.empty or len(out_df) < SEQ_LEN:
        return jsonify({'error': 'Not enough historical data for given inputs'}), 400

    # build input
    input_tensor = build_input_tensor(out_df, dt_obj).to(device)

    # inference
    with torch.no_grad():
        logit = model(input_tensor)
        prob  = torch.sigmoid(logit).item()

    return jsonify({
        'lat':        lat,
        'lon':        lon,
        'date':       dt_obj.strftime('%Y-%m-%d'),
        'prediction': prob
    })

# @app.route('/api/generateReport', methods=['GET'])
# def generateOnePager():
#     generate_report('./combined_fire_assessment.json', 'Fire_Threat_Assessment_Report.pdf')

# Simple health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/api/createEnvironment', methods=['POST'])
def create_environment():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    date = request.args.get('date')
    if lat is None or lon is None:
        return jsonify({
            'error': 'Missing required query parameters lat and lon'
        }), 400
    try:
        param = str(lon + ',' + lat)
        heightmap = generate_data_and_heightmap(param,date)
    except Exception as e:
        return jsonify({
            'error': 'Heightmap generation failed',
            'details': str(e)
        }), 500
    # Start simulation with env vars
    env = os.environ.copy()
    env["LAT"] = lat
    env["LON"] = lon
    env["DATE"] = date

    subprocess.Popen(
        ["python3", "../tactix-training/simulation.py"],
        env=env
    )
    return jsonify({
        'lat': lat,
        'lon': lon,
        'date': date
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)