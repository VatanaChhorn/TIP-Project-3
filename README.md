# COS70008-ASM3

## Description
This project is designed to analyze and process malware datasets using machine learning models. It includes preprocessing scripts, training scripts, and saved models for further analysis. The system features a hybrid approach combining Random Forest and SVM models to detect and classify various malware families with type-specific information and visual indicators.

## Project Structure
- `app/`: Contains application scripts such as `app.py`, `routes.py`, and utility functions.
  - `static/`: CSS, JavaScript, and other static assets
  - `templates/`: HTML templates for the web interface
- `data/`: Contains datasets for analysis: https://1drv.ms/f/c/88F4D600D976C0A0/EjqwFF4mgXxMmx5YkWVVx04Bd7y6rzDiRIpkU4YevoJobA?e=kbfvge
- `malimg/`
- `microsoft/`
- `models/`: Contains scripts for training and testing machine learning models.
- `models_saved/`: Contains saved models in `.pkl` format: https://1drv.ms/f/c/88F4D600D976C0A0/EmmwGnIKldxCpKt7DR7UHW4BHPw_083_UVZNnXDfVtChBw?e=MrzI6n
- `encoder_rf.pkl`
- `encoder_svm.pkl`
- `rf_model.pkl`
- `scaler_rf.pkl`
- `scaler_svm.pkl`
- `svm_model.pkl`
- `preprocessing/`: Contains preprocessing scripts for different models.

## Features
### Hybrid Malware Detection
- Combines Random Forest (static analysis) and SVM (dynamic analysis) models
- RF Model: 60% weight in decision making
- SVM Model: 40% weight in decision making

### Malware Classification System
The system classifies detected malware into specific types:

| Malware Family | Classification | Behavior |
| -------------- | -------------- | -------- |
| Ramnit | Worm / Virus / Banking Trojan | Self-replicates, steals credentials, infects executables, spreads via drives |
| Lollipop | Adware | Displays intrusive ads, tracks user behavior, hijacks browsers |
| Kelihos_ver3 | Botnet / Trojan | Sends spam, steals data, part of peer-to-peer botnet |
| Vundo | Trojan | Pop-up ads, downloads other malware, slows system |
| Simda | Backdoor Trojan | Opens system to remote control, downloads malware |
| Tracur | Trojan / Search Redirector | Redirects search results, installs additional malware |
| Kelihos_ver1 | Botnet / Trojan | Same as Kelihos_ver3, older variant |
| Obfuscator.ACY | Obfuscation Tool | Hides malware code to evade detection |
| Gatak | Trojan / Downloader | Installs additional malware, spreads via pirated software |

### Visual Indicators
- Each malware type has distinct color coding for easy identification
- Confidence levels are visually represented (high/medium/low)
- Detailed behavior explanations and recommendations based on threat level

### Reporting
- Interactive dashboard for administrators
- PDF export with comprehensive malware classification data
- Historical scan logs with detailed type information

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd COS70008-ASM3
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask server:
   ```bash
   python app/app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Upload a .bytes file for scanning
4. View the detailed analysis with malware classification

## Data
The datasets used in this project are located in the `data/` folder: 

## Models
The trained models are saved in the `models_saved/` folder as `.pkl` files:
- `encoder_rf.pkl`
- `encoder_svm.pkl`
- `rf_model.pkl`
- `scaler_rf.pkl`
- `scaler_svm.pkl`
- `svm_model.pkl`

## API Documentation
The system provides several API endpoints:

### Main Endpoints
- `/predict` (POST): Scan a file for malware
- `/api/logs` (GET): Retrieve scan history
- `/api/logs/pdf` (GET): Generate a PDF report
- `/api/malware-info` (GET): Get information about supported malware types

### Example API Response (Malware Info)
```json
{
  "Ramnit": {
    "type": "Worm / Virus / Banking Trojan",
    "behavior": "This malware could self-replicate, steal credentials, infect executables, and spread via drives."
  },
  "Kelihos_ver3": {
    "type": "Botnet / Trojan",
    "behavior": "This malware could send spam, steal your data, and add your system to a peer-to-peer botnet."
  }
  // ... other malware types
}
```

### Example API Response (Prediction)
```json
{
  "method": "Hybrid Analysis (RF + SVM)",
  "rf_result": "Kelihos_ver3",
  "rf_confidence": 96.5,
  "svm_result": "Kelihos_ver3", 
  "svm_confidence": 91.62,
  "final_result": "Kelihos_ver3",
  "final_confidence": 94.55
}
```

## Contributing
If you would like to contribute, please fork the repository and submit a pull request.

## License
Include the license information here.

## Acknowledgments
All teammates

API Testing (Postman Guide)
## Overview
The API allows users to upload .bytes (static analysis) or .png (dynamic analysis) files for Hybrid RF + SVM malware detection.

## Requirements
- Flask server running at http://127.0.0.1:5000
- Postman installed
- Prepared .bytes or .png files

## How to Test Using Postman
### 1. Start the Flask Server
python app/app.py

### 2. Create a New POST Request
- URL: http://127.0.0.1:5000/predict
- Method: POST

### 3. Configure the Request Body
- Go to Body tab in Postman
- Select form-data
- Add the following key-value pair:

Key	Type	Value
file	File	Select a .bytes or .png file

### Example
- For static: select sample.bytes
- For dynamic: select sample.png

### 4. Send the Request
## Example JSON Response
{
    "final_confidence": 0.26,
    "final_label": "6",
    "rf_confidence": 0.44,
    "rf_label": "6",
    "svm_confidence": 0.51,
    "svm_label": "Dontovo.A"
}
## Explanation of Response
- rf_label: Result from Random Forest static .bytes analysis
- rf_confidence: Confidence score for static result
- svm_label: Result from SVM dynamic .png analysis
- svm_confidence: Confidence score for dynamic result
- final_label: Hybrid weighted decision (60% RF, 40% SVM)
- final_confidence: Combined confidence score

## Notes
- Supported file types: .bytes and .png
- Form-data key must be file
- Ensure the server is running before testing
