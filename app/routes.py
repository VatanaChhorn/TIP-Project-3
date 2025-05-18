import sys
import os
import json
import datetime
import numpy as np
import cv2
from flask import Blueprint, request, jsonify, redirect, url_for, make_response
from werkzeug.utils import secure_filename
import hashlib
from flask import session, render_template
import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Global mappings for model labels and malware types
label_map = {
    "1": "Ramnit",
    "2": "Lollipop",
    "3": "Kelihos_ver3",
    "4": "Simda",
    "5": "Tracur",
    "6": "Kelihos_ver1",
    "7": "Obfuscator.ACY",
    "8": "Gatak",
    "9": "Yuner.A"
}

# Define malware types mapping - centralized for entire application
malware_types = {
    "Ramnit": "Worm / Virus / Banking Trojan",
    "Lollipop": "Adware",
    "Kelihos_ver3": "Botnet / Trojan",
    "Vundo": "Trojan",
    "Simda": "Backdoor Trojan",
    "Tracur": "Trojan / Search Redirector",
    "Kelihos_ver1": "Botnet / Trojan",
    "Obfuscator.ACY": "Obfuscation Tool",
    "Gatak": "Trojan / Downloader"
}

# Define malware behaviors
malware_behaviors = {
    "Ramnit": "This malware could self-replicate, steal credentials, infect executables, and spread via drives.",
    "Lollipop": "This malware could display intrusive ads, track your behavior, and hijack your browser.",
    "Kelihos_ver3": "This malware could send spam, steal your data, and add your system to a peer-to-peer botnet.",
    "Vundo": "This malware could display pop-up ads, download other malware, and slow down your system.",
    "Simda": "This malware could open your system to remote control and download additional malware.",
    "Tracur": "This malware could redirect your search results and install additional malware.",
    "Kelihos_ver1": "This malware could send spam, steal your data, and add your system to a peer-to-peer botnet.",
    "Obfuscator.ACY": "This malware could hide malicious code to evade detection and enable other malware.",
    "Gatak": "This malware could install additional malware and spread via pirated software."
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
        all_zeros = True
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1:  # Skip empty lines
                values = parts[1:]  # Skip address
                tokens.extend(values)
                # Check if line contains non-zero values
                if any(val != '00' for val in values):
                    all_zeros = False
        
        # If file is all zeros, return None to indicate clean file
        if all_zeros:
            return None
            
        token_counts = Counter(tokens)
        features = [len(tokens)]
        for i in range(256):
            byte_token = f'{i:02X}'
            features.append(token_counts.get(byte_token, 0))
        return features

# Helper function to convert bytes file to image features
def convert_bytes_to_image_features(bytes_path):
    """Convert .bytes file to image features for SVM model"""
    # Read .bytes file
    with open(bytes_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Convert content to a 2D representation
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
    
    return image.flatten()

# Helper function to get malware type from label
def get_malware_type(label):
    if not label or label.lower() in ('benign', 'clean', 'error', 'unknown'):
        return 'Clean'
    return malware_types.get(label, 'Unknown Malware')

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
    rf_label = ""
    rf_confidence = 0
    svm_label = ""
    svm_confidence = 0
    
    try:
        # For .bytes files
        if filename.endswith('.bytes'):
            # Get static features for RF model
            static_features = extract_features_from_bytes(file_path)
            # Generate image features for SVM model
            dynamic_features = convert_bytes_to_image_features(file_path)
        else:
            return jsonify({'error': 'Unsupported file type. Only .bytes files are supported.'}), 400

        # Process with RF model (static features)
        if static_features is not None:
            try:
                scaled_static = scaler_rf.transform([static_features])
                rf_probs = rf_model.predict_proba(scaled_static)[0]
                rf_confidence = round(np.max(rf_probs) * 100, 2)
                rf_pred = rf_model.predict(scaled_static)
                rf_class = str(encoder_rf.inverse_transform(rf_pred)[0])
                rf_label = label_map.get(rf_class, rf_class)
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
                svm_class = str(encoder_svm.inverse_transform([predicted_class_index])[0])
                svm_label = label_map.get(svm_class, svm_class)
            except Exception as e:
                print(f"SVM model error: {str(e)}")
                svm_label = "Error"
                svm_confidence = 0
        
        # Make final prediction using both models
        final_label = ""
        final_confidence = 0
        
        # Both models available - apply hybrid voting
        if rf_label and svm_label and rf_label != "Error" and svm_label != "Error":
            # Calculate weighted confidences
            rf_weighted = 0.6 * rf_confidence  # 60% weight to RF
            svm_weighted = 0.4 * svm_confidence  # 40% weight to SVM
            
            # Debug log for model comparison
            print(f"RF: {rf_label} ({rf_confidence}%), SVM: {svm_label} ({svm_confidence}%)")
            
            # If models agree on the label
            if rf_label == svm_label:
                # Use the agreed label with combined weighted confidence
                final_label = rf_label
                final_confidence = round(rf_weighted + svm_weighted, 2)
                print(f"Models agree: Final confidence = {rf_weighted} + {svm_weighted} = {final_confidence}")
            else:
                # Models disagree - use weighted confidence to determine winner
                if rf_weighted >= svm_weighted:
                    final_label = rf_label
                    final_confidence = round(rf_weighted, 2)
                else:
                    final_label = svm_label
                    final_confidence = round(svm_weighted, 2)
        
        # If only one model available or one failed, use the working model's result
        elif rf_label and rf_label != "Error":
            final_label = rf_label
            final_confidence = rf_confidence
        elif svm_label and svm_label != "Error":
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

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Error processing file'}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

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

# === API Endpoint: Export Logs as PDF ===
@routes.route('/api/logs/pdf')
def export_logs_pdf():
    # Generate a unique timestamp for this report
    report_timestamp = datetime.datetime.now()
    timestamp_str = report_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = report_timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Get username filter if provided
    username = request.args.get('username')
    
    # Clear any previous data and read directly from file for each request
    try:
        # Always read fresh from disk, never cache
        with open(SCAN_LOG_FILE, 'r') as f:
            all_logs = json.load(f)
    except Exception as e:
        print(f"Error reading scan logs: {e}")
        return jsonify({"error": f"Error reading logs: {str(e)}"}), 500
    
    # Filter logs by username if needed
    if username:
        logs = [log for log in all_logs if log.get('username') == username]
        report_title = f"Malware Scan Report - User: {username}"
        filename = f"malware_scan_report_{username}_{file_timestamp}.pdf"
    else:
        logs = all_logs
        report_title = "Malware Scan Report - All Users"
        filename = f"malware_scan_report_all_{file_timestamp}.pdf"
    
    # Count malware vs clean files
    malware_count = sum(1 for log in logs if log.get('final_label', '').lower() not in ('benign', 'clean'))
    benign_count = sum(1 for log in logs if log.get('final_label', '').lower() in ('benign', 'clean'))
    total_scans = len(logs)
    
    # Debug info
    print(f"[PDF] Generating report at {timestamp_str}")
    print(f"[PDF] Total logs found: {total_scans}")
    print(f"[PDF] Username filter: {username if username else 'None'}")
    
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Set up the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center alignment
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        alignment=1  # Center alignment
    )
    
    # Create table data
    data = [
        ['Filename', 'Timestamp', 'RF Label', 'RF Conf', 'SVM Label', 'SVM Conf', 'Final Label', 'Final Conf', 'Malware Type']
    ]
    
    # Add rows to table in reverse timestamp order (newest first)
    sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    for log in sorted_logs:
        # Get malware type if available
        final_label = log.get('final_label', '')
        malware_type = get_malware_type(final_label)
            
        row = [
            log.get('filename', ''),
            log.get('timestamp', '')[:16] if log.get('timestamp') else '',
            log.get('rf_label', ''),
            f"{log.get('rf_confidence', '')}%" if log.get('rf_confidence') not in ('', None) else '',
            log.get('svm_label', ''),
            f"{log.get('svm_confidence', '')}%" if log.get('svm_confidence') not in ('', None) else '',
            log.get('final_label', ''),
            f"{log.get('final_confidence', '')}%" if log.get('final_confidence') not in ('', None) else '',
            malware_type
        ]
        data.append(row)
    
    # Create table with clear style
    table = Table(data, repeatRows=1)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
    ])
    table.setStyle(table_style)
    
    # Build the PDF content
    elements = []
    
    # Add title and timestamp
    elements.append(Paragraph(report_title, title_style))
    elements.append(Paragraph(f"Generated on: {timestamp_str}", subtitle_style))
    
    # Add summary
    summary_style = styles['Normal']
    summary_style.spaceAfter = 12
    
    elements.append(Spacer(1, 0.25 * inch))
    elements.append(Paragraph(f"Total Scans: {total_scans}", summary_style))
    elements.append(Paragraph(f"Malware Detected: {malware_count}", summary_style))
    elements.append(Paragraph(f"Clean Files: {benign_count}", summary_style))
    elements.append(Spacer(1, 0.25 * inch))
    
    # Add table
    elements.append(table)
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Create response with cache prevention headers
    response = make_response(pdf_data)
    response.headers["Content-Disposition"] = f"inline; filename={filename}"
    response.headers["Content-Type"] = "application/pdf"
    
    # Add cache prevention headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    print(f"[PDF] Report generated successfully with {len(sorted_logs)} records")
    return response

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
        return jsonify({
            "message": "Login successful.", 
            "role": role,
            "username": username
        }), 200

    return jsonify({"message": "Invalid username or password."}), 401

@routes.route('/logout')
def logout():
    # Clear the session
    session.clear()
    # Redirect to home page
    return redirect('/')

# === API Endpoint: Get Malware Info ===
@routes.route('/api/malware-info')
def get_malware_info():
    """Returns malware types and behaviors for frontend use"""
    malware_info = {}
    for malware, malware_type in malware_types.items():
        malware_info[malware] = {
            "type": malware_type,
            "behavior": malware_behaviors.get(malware, "Unknown behavior")
        }
    return jsonify(malware_info)