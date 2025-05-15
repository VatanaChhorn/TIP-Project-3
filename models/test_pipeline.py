import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def process_input_data(input_file):
    data = pd.read_csv(input_file)
    return data

def make_prediction(model, data):
    predictions = model.predict(data)
    return predictions

def main(input_file, model_path):
    model = load_model(model_path)
    data = process_input_data(input_file)
    predictions = make_prediction(model, data)
    print(predictions)

if __name__ == "__main__":
    input_file = 'path/to/uploaded/file.csv'
    model_path = 'path/to/model.pkl'
    main(input_file, model_path)