import os
import bcrypt
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required,
    get_jwt_identity, get_jwt
)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import numpy as np

# Optional TensorFlow imports
MODEL = None
cv2 = None
load_model = None

try:
    from tensorflow.keras.models import load_model
    import cv2
except Exception as e:
    print("TensorFlow/OpenCV not available:", e)

# Config
DB_NAME = 'tumor_db'
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
JWT_SECRET = 'change-this-secret'

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = JWT_SECRET
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/tumor_db'

db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum('patient', 'doctor'), nullable=False, default='patient')
    created_at = db.Column(db.DateTime, default=db.func.now())


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(255))
    age = db.Column(db.Integer)
    gender = db.Column(db.Enum('male', 'female', 'other'))
    contact = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=db.func.now())
    user = db.relationship('User', backref=db.backref('patient_profile', uselist=False))


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    image_path = db.Column(db.String(1024))
    reviewed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.now())
    patient = db.relationship('Patient', backref='predictions')


# Load optional model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'tumor_model.h5')
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = 150


def try_load_model():
    global MODEL

    if load_model is None:
        app.logger.warning("TensorFlow not available, ML disabled")
        return

    if os.path.exists(MODEL_PATH):
        try:
            MODEL = load_model(MODEL_PATH)
            app.logger.info("Model loaded successfully")
        except Exception as e:
            app.logger.error(f"Model load failed: {e}")
            MODEL = None
    else:
        app.logger.warning("Model file not found")


try_load_model()


def preprocess_image_bytes(image_bytes):
    if cv2 is None:
        raise RuntimeError('OpenCV/TensorFlow not available.')

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError('Invalid image file')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, 0)
    return img


# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


# Registration
def bcrypt_hash(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')


@app.route('/register', methods=['POST'])
def register():
    data = request.json or {}
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'patient')

    if not username or not email or not password:
        return jsonify({'status': 'error', 'msg': 'username, email and password required'}), 400

    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'status': 'error', 'msg': 'user exists'}), 400

    hashed = bcrypt_hash(password)

    user = User(username=username, email=email, password=hashed, role=role)
    db.session.add(user)
    db.session.flush()

    if role == 'patient':
        patient = Patient(user_id=user.id, name=username)
        db.session.add(patient)

    db.session.commit()
    return jsonify({'status': 'success', 'msg': 'registered'}), 201


# Login
def bcrypt_check(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


@app.route('/login', methods=['POST'])
def login():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'status': 'error', 'msg': 'username and password required'}), 400

    user = User.query.filter((User.username == username) | (User.email == username)).first()

    if user and bcrypt_check(password, user.password):
        token = create_access_token(
            identity=str(user.id),
            additional_claims={'role': user.role},
            expires_delta=datetime.timedelta(hours=8)
        )
        return jsonify({'status': 'success', 'access_token': token, 'role': user.role}), 200

    return jsonify({'status': 'error', 'msg': 'invalid credentials'}), 401


# Upload scan
@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_scan():
    user_id = int(get_jwt_identity())
    role = get_jwt().get('role')

    patient_id = request.form.get('patient_id')

    if role == 'patient':
        patient = Patient.query.filter_by(user_id=user_id).first()
        if not patient:
            return jsonify({'status': 'error', 'msg': 'patient profile not found'}), 400

    elif role == 'doctor':
        if not patient_id:
            return jsonify({'status': 'error', 'msg': 'patient_id is required for doctor uploads'}), 400

        patient = Patient.query.filter_by(id=int(patient_id)).first()
        if not patient:
            return jsonify({'status': 'error', 'msg': 'patient not found'}), 404
    else:
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'msg': 'image required'}), 400

    f = request.files['image']
    fname = secure_filename(f.filename)
    ts = int(datetime.datetime.utcnow().timestamp())
    save_name = f"{patient.id}_{ts}_{fname}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    f.save(save_path)

    if MODEL:
        try:
            img_bytes = open(save_path, 'rb').read()
            arr = preprocess_image_bytes(img_bytes)
            preds = MODEL.predict(arr)[0]
            idx = int(np.argmax(preds))
            label = CLASS_LABELS[idx]
            conf = float(np.max(preds))
        except Exception:
            label = 'Error'
            conf = 0.0
    else:
        label = 'No Model'
        conf = 0.0

    pred = Prediction(
        patient_id=patient.id,
        label=label,
        confidence=conf,
        image_path=save_name
    )

    db.session.add(pred)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'prediction': {
            'id': pred.id,
            'label': label,
            'confidence': conf,
            'image_url': f"/uploads/{save_name}"
        }
    })


# List all patients for doctor
@app.route('/doctor/patients', methods=['GET'])
@jwt_required()
def list_patients():
    if get_jwt().get('role') != 'doctor':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    patients = Patient.query.all()
    data = []

    for p in patients:
        data.append({
            'id': p.id,
            'name': p.name,
            'age': p.age,
            'gender': p.gender,
            'contact': p.contact
        })

    return jsonify({'status': 'success', 'patients': data})


# Patient reports
@app.route('/patient/reports', methods=['GET'])
@jwt_required()
def my_reports():
    user_id = int(get_jwt_identity())

    if get_jwt().get('role') != 'patient':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    patient = Patient.query.filter_by(user_id=user_id).first()
    if not patient:
        return jsonify({'status': 'error', 'msg': 'patient not found'}), 400

    reports = []
    for p in patient.predictions:
        reports.append({
            'id': p.id,
            'label': p.label,
            'confidence': p.confidence,
            'image_url': f"/uploads/{os.path.basename(p.image_path)}",
            'reviewed': p.reviewed,
            'created_at': p.created_at.isoformat()
        })

    return jsonify({'status': 'success', 'reports': reports})


# Create or update patient profile
@app.route('/patient/profile', methods=['POST'])
@jwt_required()
def create_or_update_profile():
    user_id = int(get_jwt_identity())

    if get_jwt().get('role') != 'patient':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    data = request.json or {}
    name = data.get('name')
    age = data.get('age')
    gender = data.get('gender')
    contact = data.get('contact')

    if not name or not age or not gender or not contact:
        return jsonify({'status': 'error', 'msg': 'all fields are required'}), 400

    patient = Patient.query.filter_by(user_id=user_id).first()

    if patient:
        patient.name = name
        patient.age = age
        patient.gender = gender
        patient.contact = contact
        msg = 'Profile updated successfully'
    else:
        patient = Patient(user_id=user_id, name=name, age=age, gender=gender, contact=contact)
        db.session.add(patient)
        msg = 'Profile created successfully'

    db.session.commit()
    return jsonify({'status': 'success', 'msg': msg})


# Doctor reports
@app.route('/doctor/reports', methods=['GET'])
@jwt_required()
def all_reports():
    if get_jwt().get('role') != 'doctor':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    preds = Prediction.query.order_by(Prediction.created_at.desc()).all()
    reports = []

    for p in preds:
        reports.append({
            'id': p.id,
            'patient_id': p.patient_id,
            'label': p.label,
            'confidence': p.confidence,
            'image_url': f"/uploads/{os.path.basename(p.image_path)}",
            'reviewed': p.reviewed,
            'created_at': p.created_at.isoformat()
        })

    return jsonify({'status': 'success', 'reports': reports})


# Doctor review
@app.route('/doctor/review/<int:pred_id>', methods=['POST'])
@jwt_required()
def review_prediction(pred_id):
    if get_jwt().get('role') != 'doctor':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    pred = Prediction.query.get(pred_id)
    if not pred:
        return jsonify({'status': 'error', 'msg': 'not found'}), 404

    pred.reviewed = True
    db.session.commit()
    return jsonify({'status': 'success', 'msg': 'marked reviewed'})


# Delete report
@app.route('/doctor/reports/<int:report_id>', methods=['DELETE'])
@jwt_required()
def delete_report(report_id):
    if get_jwt().get('role') != 'doctor':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    pred = Prediction.query.get(report_id)
    if not pred:
        return jsonify({'status': 'error', 'msg': 'report not found'}), 404

    image_path = os.path.join(UPLOAD_DIR, os.path.basename(pred.image_path or ''))
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception as e:
            app.logger.warning(f"Failed to delete image file {image_path}: {e}")

    db.session.delete(pred)
    db.session.commit()
    return jsonify({'status': 'success', 'msg': 'report deleted successfully'})


@app.route('/doctor/patient/<int:patient_id>', methods=['GET'])
@jwt_required()
def get_patient(patient_id):
    if get_jwt().get('role') != 'doctor':
        return jsonify({'status': 'error', 'msg': 'forbidden'}), 403

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({'status': 'error', 'msg': 'patient not found'}), 404

    return jsonify({
        'status': 'success',
        'patient': {
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'gender': patient.gender,
            'contact': patient.contact
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
