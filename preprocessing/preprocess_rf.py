import os
import csv
from collections import Counter

# Path to .bytes files folder
BYTES_FOLDER = '../data/microsoft/train/'
OUTPUT_FILE = '../data/microsoft/static_features.csv'

def process_file(filepath):
    with open(filepath, 'r', errors='ignore') as file:
        tokens = []
        for line in file:
            parts = line.strip().split()
            # Skip the memory address part (first column)
            tokens.extend(parts[1:])
    return tokens

def extract_features():
    rows = []
    headers = ['Id', 'Size'] + [f'Byte_{i:02X}' for i in range(256)]
    
    for filename in os.listdir(BYTES_FOLDER):
        if not filename.endswith('.bytes'):
            continue
        file_id = filename.replace('.bytes', '')
        filepath = os.path.join(BYTES_FOLDER, filename)
        tokens = process_file(filepath)
        
        token_counts = Counter(tokens)
        row = [file_id, len(tokens)]
        
        for i in range(256):
            byte_token = f'{i:02X}'
            row.append(token_counts.get(byte_token, 0))
        
        rows.append(row)
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == "__main__":
    extract_features()
    print(f"Feature extraction complete. CSV saved to {OUTPUT_FILE}")
