# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import os
import sys
import json

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ── Config ──────────────────────────────────────────────

# FIX 2: Harden SECRET_KEY — crash loudly in production if not set
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production':
        print("ERROR: SECRET_KEY environment variable is not set in production!", file=sys.stderr)
        sys.exit(1)
    else:
        secret_key = 'dev-only-insecure-key'  # local development only

app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    f"sqlite:///{os.path.join(BASE_DIR, 'elcare.db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Extensions ───────────────────────────────────────────

CORS(app)

from models import db, User, PredictionHistory
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# FIX 1: Return JSON 401 instead of HTML redirect for unauthenticated API calls
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'success': False, 'message': 'Login required'}), 401

# ── Blueprints ────────────────────────────────────────────

from auth import auth_bp
from profile_routes import profile_bp

app.register_blueprint(auth_bp)
app.register_blueprint(profile_bp)

# ── ML Models ─────────────────────────────────────────────

class MedicalPredictor:
    def __init__(self):
        self.parkinsons_model = None
        self.heart_model = None
        self.diabetes_model = None
        self.parkinsons_scaler = None
        self.diabetes_scaler = None
        self.train_models()

    def train_parkinsons_model(self):
        try:
            parkinsons_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'parkinsons.data'))
            X = parkinsons_data.drop(columns=['name', 'status'])
            Y = parkinsons_data['status']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            self.parkinsons_scaler = scaler
            model = svm.SVC(kernel='linear', probability=True)
            model.fit(X_train, Y_train)
            self.parkinsons_model = model
            return {
                "status": "success",
                "train_accuracy": model.score(X_train, Y_train),
                "test_accuracy": model.score(scaler.transform(X_test), Y_test)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def train_heart_model(self):
        try:
            heart_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'heart_disease_data.csv'))
            X = heart_data.drop(columns='target')
            Y = heart_data['target']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
            model = LogisticRegression()
            model.fit(X_train, Y_train)
            self.heart_model = model
            return {
                "status": "success",
                "train_accuracy": model.score(X_train, Y_train),
                "test_accuracy": model.score(X_test, Y_test)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def train_diabetes_model(self):
        try:
            diabetes_dataset = pd.read_csv(os.path.join(BASE_DIR, 'data', 'diabetes (2).csv'))
            X = diabetes_dataset.drop(columns='Outcome')
            Y = diabetes_dataset['Outcome']
            scaler = StandardScaler()
            scaler.fit(X)
            standardized_data = scaler.transform(X)
            self.diabetes_scaler = scaler
            X_train, X_test, Y_train, Y_test = train_test_split(
                standardized_data, Y, test_size=0.2, stratify=Y, random_state=2
            )
            model = svm.SVC(kernel='linear', probability=True)
            model.fit(X_train, Y_train)
            self.diabetes_model = model
            return {
                "status": "success",
                "train_accuracy": model.score(X_train, Y_train),
                "test_accuracy": model.score(X_test, Y_test)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def train_models(self):
        print("Training Parkinson's model...")
        print(self.train_parkinsons_model())
        print("Training Heart model...")
        print(self.train_heart_model())
        print("Training Diabetes model...")
        print(self.train_diabetes_model())

predictor = MedicalPredictor()

# ── Helper: validate prediction input ────────────────────

# FIX 3: Input validation helper — prevents crashes from malformed requests
def validate_features(data, expected_count):
    """Returns (features_list, error_message). error is None if valid."""
    if not data or 'features' not in data:
        return None, "Missing 'features' key in request body"

    features = data['features']

    if not isinstance(features, list):
        return None, "'features' must be a list"

    if len(features) != expected_count:
        return None, f"Expected {expected_count} features, got {len(features)}"

    try:
        features = [float(f) for f in features]
    except (TypeError, ValueError):
        return None, "All features must be numeric values"

    return features, None

# ── Helper: save prediction to DB ────────────────────────

def save_prediction(disease_type, features, result, message):
    if current_user.is_authenticated:
        try:
            record = PredictionHistory(
                user_id=current_user.id,
                disease_type=disease_type,
                input_features=json.dumps(features),
                prediction_result=result,
                result_message=message
            )
            db.session.add(record)
            db.session.commit()
        except Exception as e:
            print(f"Failed to save prediction: {e}")

# ── Page Routes ──────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/parkinsons')
def parkinsons_page():
    return render_template('parkinsons.html')

@app.route('/heart')
def heart_page():
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

# ── API: Predictions ─────────────────────────────────────

# FIX 1: Added @login_required to all three prediction endpoints
# FIX 3: Added validate_features() call before processing

@app.route('/api/predict/parkinsons', methods=['POST'])
@login_required
def predict_parkinsons():
    features, error = validate_features(request.json, expected_count=22)
    if error:
        return jsonify({"error": error}), 400

    try:
        input_data = np.asarray(features).reshape(1, -1)
        if not predictor.parkinsons_scaler:
            return jsonify({"error": "Scaler not loaded"}), 500
        std_data = predictor.parkinsons_scaler.transform(input_data)
        if not predictor.parkinsons_model:
            return jsonify({"error": "Model not loaded"}), 500
        prediction = predictor.parkinsons_model.predict(std_data)
        result = int(prediction[0])
        message = "Parkinson's Disease Detected" if result == 1 else "No Parkinson's Disease Detected"
        save_prediction('parkinsons', features, result, message)
        return jsonify({"prediction": result, "message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/heart', methods=['POST'])
@login_required
def predict_heart():
    features, error = validate_features(request.json, expected_count=13)
    if error:
        return jsonify({"error": error}), 400

    try:
        input_data = np.asarray(features).reshape(1, -1)
        if not predictor.heart_model:
            return jsonify({"error": "Model not loaded"}), 500
        prediction = predictor.heart_model.predict(input_data)
        result = int(prediction[0])
        message = "Heart Disease Detected" if result == 1 else "No Heart Disease Detected"
        save_prediction('heart', features, result, message)
        return jsonify({"prediction": result, "message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    features, error = validate_features(request.json, expected_count=8)
    if error:
        return jsonify({"error": error}), 400

    try:
        input_data = np.asarray(features).reshape(1, -1)
        if not predictor.diabetes_scaler:
            return jsonify({"error": "Scaler not loaded"}), 500
        std_data = predictor.diabetes_scaler.transform(input_data)
        if not predictor.diabetes_model:
            return jsonify({"error": "Model not loaded"}), 500
        prediction = predictor.diabetes_model.predict(std_data)
        result = int(prediction[0])
        message = "Diabetes Detected" if result == 1 else "No Diabetes Detected"
        save_prediction('diabetes', features, result, message)
        return jsonify({"prediction": result, "message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        "parkinsons": {"name": "Parkinson's Disease Predictor", "features": 22, "model_type": "SVM with Linear Kernel"},
        "heart":      {"name": "Heart Disease Predictor",       "features": 13, "model_type": "Logistic Regression"},
        "diabetes":   {"name": "Diabetes Predictor",            "features": 8,  "model_type": "SVM with Linear Kernel"}
    })

# ── DB Init ───────────────────────────────────────────────

with app.app_context():
    db.create_all()
    print("Database tables created.")

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    app.run(debug=True, port=5000)