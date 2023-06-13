import pandas as pd
import shutil
import os

csv_cancer=pd.read_csv("zr7vgbcyr2-1//metadata.csv")

def move_images(data, source_folder, destination_parent_folder):
    unique_diagnoses = data["diagnostic"].unique()

    # Create destination folders for each unique diagnosis
    for diagnosis in unique_diagnoses:
        destination_folder = os.path.join(destination_parent_folder, diagnosis)
        os.makedirs(destination_folder, exist_ok=True)

    # Move files to the respective destination folders
    for index, row in data.iterrows():
        img_id = row['img_id']
        diagnostic = row['diagnostic']
        file_name = img_id
        source_path = os.path.join(source_folder, file_name)
        destination_folder = os.path.join(destination_parent_folder, diagnostic)
        destination_path = os.path.join(destination_folder, file_name)

        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved {file_name} to {destination_folder}/{file_name}")
        else:
            print(f"File {file_name} not found in {source_folder}")