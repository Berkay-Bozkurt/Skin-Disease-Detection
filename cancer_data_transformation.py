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


