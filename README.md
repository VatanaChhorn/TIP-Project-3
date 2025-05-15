# COS70008-ASM3

## Description
This project is designed to analyze and process malware datasets using machine learning models. It includes preprocessing scripts, training scripts, and saved models for further analysis.

## Project Structure
- `app/`: Contains application scripts such as `app.py`, `routes.py`, and utility functions.
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
1. Preprocess the data using the scripts in the `preprocessing/` folder.
2. Train the models using the scripts in the `models/` folder.
3. Use the saved models in `models_saved/` for predictions.

## Data
The datasets used in this project are located in the `data/` folder: https://1drv.ms/f/c/88F4D600D976C0A0/EjqwFF4mgXxMmx5YkWVVx04Bd7y6rzDiRIpkU4YevoJobA?e=kbfvge
- `malimg/`
- `microsoft/`

## Models
The trained models are saved in the `models_saved/` folder as `.pkl` files: https://1drv.ms/f/c/88F4D600D976C0A0/EmmwGnIKldxCpKt7DR7UHW4BHPw_083_UVZNnXDfVtChBw?e=MrzI6n
- `encoder_rf.pkl`
- `encoder_svm.pkl`
- `rf_model.pkl`
- `scaler_rf.pkl`
- `scaler_svm.pkl`
- `svm_model.pkl`

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
