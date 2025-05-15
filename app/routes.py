import sys
import os
import json
import datetime
import numpy as np
import cv2
from flask import Blueprint, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import hashlib
from flask import session, render_template
import io
from fpdf import FPDF
from flask import make_response

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

label_map = {
    "0": "Ramnit",
    "1": "Lollipop",
    "2": "Kelihos_ver3",
    "3": "Vundo",
    "4": "Simda",
    "5": "Tracur",
    "6": "Kelihos_ver1",
    "7": "Obfuscator.ACY",
    "8": "Gatak"
}

# Add models folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from utils import load_models

routes = Blueprint('routes', __name__)

# === Load Trained Models ===
rf_model, svm_model, scaler_rf, scaler_svm, encoder_rf, encoder_svm = load_models()

# === User Data File ===
USER_FILE = 'users.json'

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, 'r') as f:
        users = json.load(f)
        return {user['username']: user for user in users}

def save_users(user_store):
    with open(USER_FILE, 'w') as f:
        json.dump(list(user_store.values()), f, indent=2)

# Load users on server start
user_store = load_users()

# === Temp Folder for Uploaded Files ===
UPLOAD_FOLDER = './temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Scan Log File ===
SCAN_LOG_FILE = 'scan_logs.json'

def save_scan_log(log_entry):
    if not os.path.exists(SCAN_LOG_FILE):
        logs = []
    else:
        with open(SCAN_LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(log_entry)
    with open(SCAN_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

# === Static Feature Extraction ===
def extract_features_from_bytes(file_path):
    from collections import Counter
    with open(file_path, 'r', errors='ignore') as file:
        tokens = []
        for line in file:
            parts = line.strip().split()
            tokens.extend(parts[1:])
    token_counts = Counter(tokens)
    features = [len(tokens)]
    for i in range(256):
        byte_token = f'{i:02X}'
        features.append(token_counts.get(byte_token, 0))
    return features

@routes.route('/api/logs/pdf')
def get_logs_pdf():
    username = request.args.get('username')
    if not os.path.exists(SCAN_LOG_FILE):
        return jsonify({'error': 'No logs found'}), 404
    
    with open(SCAN_LOG_FILE, 'r') as f:
        logs = json.load(f)
    
    if username:
        logs = [log for log in logs if log['username'] == username]
        report_title = f"Scan Report for User: {username}"
    else:
        report_title = "Complete Scan Report for All Users"
    
    # Create PDF with FPDF - ensure landscape orientation for better table display
    pdf = FPDF(orientation='L')  # 'L' for landscape orientation
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, report_title, 0, 1, 'C')
    pdf.ln(5)
    
    # Add timestamp
    pdf.set_font('Arial', '', 10)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated: {current_time}", 0, 1, 'R')
    pdf.ln(5)
    
    # Recalculate column widths for landscape mode - make proportional
    total_width = pdf.w - 20  # Total width minus margins
    col_widths = [
        total_width * 0.28,  # Filename (28%)
        total_width * 0.14,  # Timestamp (14%)
        total_width * 0.12,  # RF Label (12%)
        total_width * 0.08,  # RF Conf (8%)
        total_width * 0.12,  # SVM Label (12%)
        total_width * 0.08,  # SVM Conf (8%)
        total_width * 0.12,  # Final Label (12%)
        total_width * 0.08   # Final Conf (8%)
    ]
    
    # Add table headers
    pdf.set_font('Arial', 'B', 10)
    headers = ['Filename', 'Timestamp', 'RF Label', 'RF Conf', 'SVM Label', 'SVM Conf', 'Final Label', 'Final Conf']
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()
    
    # Add table data with fixed cell heights
    pdf.set_font('Arial', '', 9)
    for log in logs:
        # Limit filename length to prevent overflow
        filename = log.get('filename', '')
        if len(filename) > 40:
            filename = filename[:37] + '...'
            
        pdf.cell(col_widths[0], 8, filename, 1)
        pdf.cell(col_widths[1], 8, log.get('timestamp', '')[:16], 1)  # Trim timestamp
        pdf.cell(col_widths[2], 8, log.get('rf_label', ''), 1)
        pdf.cell(col_widths[3], 8, str(log.get('rf_confidence', '')), 1)
        pdf.cell(col_widths[4], 8, log.get('svm_label', ''), 1)
        pdf.cell(col_widths[5], 8, str(log.get('svm_confidence', '')), 1)
        pdf.cell(col_widths[6], 8, log.get('final_label', ''), 1)
        pdf.cell(col_widths[7], 8, str(log.get('final_confidence', '')), 1)
        pdf.ln()
    
    # Summary statistics - use more space for malware distribution
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Total Scans: {len(logs)}", 0, 1)
    
    if len(logs) > 0:
        # Count by malware type
        malware_counts = {}
        for log in logs:
            label = log.get('final_label', 'Unknown')
            if label:
                malware_counts[label] = malware_counts.get(label, 0) + 1
        
        pdf.cell(0, 10, "Malware Distribution:", 0, 1)
        
        # Improved layout for malware distribution - use even more space in landscape
        items_per_column = 3
        col_width = 90  # Wider columns in landscape mode
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        for i, (label, count) in enumerate(malware_counts.items()):
            percentage = (count / len(logs)) * 100
            col = i // items_per_column
            row = i % items_per_column
            
            # Position for this item
            pdf.set_xy(x_start + (col * col_width), y_start + (row * 8))
            pdf.cell(col_width, 8, f"- {label}: {count} ({percentage:.1f}%)", 0)
        
        # Reset position after the malware distribution
        max_rows = (len(malware_counts) + items_per_column - 1) // items_per_column
        pdf.set_y(y_start + (max_rows * 8) + 10)
    
    # Generate PDF with proper encoding
    pdf_output = io.BytesIO()
    pdf_string = pdf.output(dest='S')
    pdf_output.write(pdf_string.encode('latin-1'))
    pdf_output.seek(0)
    
    # Create response
    response = make_response(pdf_output.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=scan_report{"_"+username if username else ""}.pdf'
    
    return response

# === File Scan Endpoint with True Hybrid Mode ===
@routes.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)

    # Variables to store features and predictions
    static_features = None
    dynamic_features = None
    rf_label = ""
    rf_confidence = 0
    svm_label = ""
    svm_confidence = 0
    
    # Get static features (.bytes analysis)
    if filename.endswith('.bytes'):
        static_features = extract_features_from_bytes(file_path)
        # Convert to image for dynamic analysis
        try:
            # Create visualization from bytes for SVM model
            dynamic_features = convert_bytes_to_image_features(file_path)
        except Exception as e:
            print(f"Error generating image features: {str(e)}")
            dynamic_features = None
    
    # Get dynamic features (.png analysis)
    elif filename.endswith('.png'):
        # Load image for SVM
        try:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, (64, 64))
                dynamic_features = resized_image.flatten()
            
            # Extract byte patterns from image for RF
            try:
                static_features = extract_features_from_image(file_path)
            except Exception as e:
                print(f"Error extracting static features: {str(e)}")
                static_features = None
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Invalid image file'}), 400
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    # Process with RF model (static features)
    if static_features is not None:
        try:
            scaled_static = scaler_rf.transform([static_features])
            rf_probs = rf_model.predict_proba(scaled_static)[0]
            rf_confidence = round(np.max(rf_probs) * 100, 2)
            rf_pred = rf_model.predict(scaled_static)
            rf_label = str(encoder_rf.inverse_transform(rf_pred)[0])
            rf_label = label_map.get(rf_label, rf_label)
        except Exception as e:
            print(f"RF model error: {str(e)}")
            rf_label = "Error"
            rf_confidence = 0
    
    # Process with SVM model (dynamic features)
    if dynamic_features is not None:
        try:
            scaled_dynamic = scaler_svm.transform([dynamic_features])
            svm_probs = svm_model.predict_proba(scaled_dynamic)[0]
            svm_confidence = round(np.max(svm_probs) * 100, 2)
            predicted_class_index = np.argmax(svm_probs)
            svm_label = str(encoder_svm.inverse_transform([predicted_class_index])[0])
        except Exception as e:
            print(f"SVM model error: {str(e)}")
            svm_label = "Error"
            svm_confidence = 0
    
    # Make final prediction using both models if available
    final_label = ""
    final_confidence = 0
    
    # Both models available - apply hybrid voting
    if rf_label and svm_label:
        # Apply weighted voting (60% RF, 40% SVM)
        if rf_label == svm_label:
            # If models agree, use the label with combined confidence
            final_label = rf_label
            final_confidence = round(0.6 * rf_confidence + 0.4 * svm_confidence, 2)
        else:
            # Models disagree, use weighted confidence to determine winner
            rf_weighted = 0.6 * rf_confidence
            svm_weighted = 0.4 * svm_confidence
            
            if rf_weighted >= svm_weighted:
                final_label = rf_label
                final_confidence = round(rf_weighted, 2)
            else:
                final_label = svm_label
                final_confidence = round(svm_weighted, 2)
    
    # If only one model available, use its result
    elif rf_label:
        final_label = rf_label
        final_confidence = rf_confidence
    elif svm_label:
        final_label = svm_label
        final_confidence = svm_confidence
    else:
        final_label = "Unknown"
        final_confidence = 0
    
    # Save log
    if 'username' in session:
        log_entry = {
            'username': session['username'],
            'filename': filename,
            'timestamp': str(datetime.datetime.now()),
            'rf_label': rf_label,
            'rf_confidence': rf_confidence,
            'svm_label': svm_label,
            'svm_confidence': svm_confidence,
            'final_label': final_label,
            'final_confidence': final_confidence
        }
        save_scan_log(log_entry)

    return jsonify({
        'method': 'Hybrid Analysis (RF + SVM)',
        'rf_result': rf_label,
        'rf_confidence': rf_confidence,
        'svm_result': svm_label,
        'svm_confidence': svm_confidence,
        'final_result': final_label,
        'final_confidence': final_confidence
    }), 200

# Helper function to extract static features from an image file
def extract_features_from_image(image_path):
    # Read binary content from image file
    with open(image_path, 'rb') as f:
        binary_data = f.read()
    
    # Convert binary to hex representation
    hex_data = binary_data.hex()
    
    # Create byte frequency features (similar to .bytes files)
    features = [len(hex_data) // 2]  # Length of binary data
    byte_counts = {}
    for i in range(0, len(hex_data), 2):
        if i + 1 < len(hex_data):
            byte = hex_data[i:i+2].upper()
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
    
    # Create feature vector with same structure as RF model expects
    for i in range(256):
        byte_token = f'{i:02X}'
        features.append(byte_counts.get(byte_token, 0))
    
    return features

# Helper function to convert bytes file to image features
def convert_bytes_to_image_features(bytes_path):
    # Read .bytes file
    with open(bytes_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Convert content to a 2D representation
    # We'll create a simple grayscale image using byte values
    bytes_values = []
    for line in content.split('\n'):
        parts = line.strip().split()
        if len(parts) > 1:  # Skip empty lines
            row = []
            for byte in parts[1:]:  # Skip address
                try:
                    value = int(byte, 16)
                    row.append(value)
                except ValueError:
                    row.append(0)  # Use 0 for invalid values
            bytes_values.extend(row)
    
    # Create a square image
    size = int(np.sqrt(len(bytes_values)))
    if size < 64:
        size = 64  # Minimum size
    
    # Pad or truncate to size²
    if len(bytes_values) < size * size:
        bytes_values.extend([0] * (size * size - len(bytes_values)))
    elif len(bytes_values) > size * size:
        bytes_values = bytes_values[:size * size]
    
    # Reshape and resize to 64x64
    image = np.array(bytes_values).reshape(size, size).astype(np.uint8)
    image = cv2.resize(image, (64, 64))
    
    # Return flattened features
    return image.flatten()

    return jsonify({'error': 'Unsupported file type'}), 400

# === API Endpoint: Get Scan Logs ===
@routes.route('/api/logs')
def get_logs():
    username = request.args.get('username')
    if not os.path.exists(SCAN_LOG_FILE):
        return jsonify([])
    with open(SCAN_LOG_FILE, 'r') as f:
        logs = json.load(f)
    if username:
        logs = [log for log in logs if log['username'] == username]
    return jsonify(logs)

# === User Registration Endpoint ===
@routes.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    fullname = data.get('fullname')
    role = data.get('role', 'user')  # Default to 'user' if not provided

    if not username or not password or not email or not fullname:
        return jsonify({'message': 'All fields are required.'}), 400

    if username in user_store:
        return jsonify({'message': 'Username already exists.'}), 409

    user_store[username] = {
        'fullname': fullname,
        'email': email,
        'username': username,
        'password': hash_password(password),
        'role': role,
        'created_at': str(datetime.datetime.now())
    }

    save_users(user_store)
    return jsonify({'message': f'Registration successful as {role}.'}), 200


@routes.route('/admin')
def admin_dashboard():
    if 'role' not in session or session['role'] != 'admin':
        return render_template('access_denied.html'), 403
    return render_template('admin.html')

# === User Management Endpoint ===
@routes.route('/api/users')
def get_users():
    # Return all usernames
    users = []
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            user_data = json.load(f)
            users = [user['username'] for user in user_data]
    return jsonify(users)

# === User Login Endpoint ===
@routes.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = user_store.get(username)
    if user and user['password'] == hash_password(password):
        session['username'] = username
        session['role'] = user.get('role', 'user')
        role = user.get('role', 'user')
        return jsonify({"message": "Login successful.", "role": role}), 200

    return jsonify({"message": "Invalid username or password."}), 401

@routes.route('/logout')
def logout():
    # Clear the session
    session.clear()
    # Redirect to home page
    return redirect('/')