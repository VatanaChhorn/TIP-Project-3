import os
import csv
from PIL import Image
import numpy as np

# Path to the malimg image folders
MALIMG_FOLDER = '../data/malimg'
# Path to save the output CSV file
OUTPUT_FILE = '../data/malimg/image_features.csv'

# Resize all images to the same size (e.g., 64x64)
IMAGE_SIZE = (64, 64)

def process_image(image_path):
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize(IMAGE_SIZE)
    return np.array(img).flatten()

def extract_features():
    rows = []
    headers = ['FileID'] + [f'Pixel_{i}' for i in range(IMAGE_SIZE[0] * IMAGE_SIZE[1])] + ['Label']

    for malware_family in os.listdir(MALIMG_FOLDER):
        folder_path = os.path.join(MALIMG_FOLDER, malware_family)
        if not os.path.isdir(folder_path):
            continue

        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            file_id = f"{malware_family}_{image_file}"
            image_path = os.path.join(folder_path, image_file)
            pixel_array = process_image(image_path).tolist()
            row = [file_id] + pixel_array + [malware_family]
            rows.append(row)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == "__main__":
    extract_features()
    print(f"Image feature extraction complete. CSV saved to {OUTPUT_FILE}")
