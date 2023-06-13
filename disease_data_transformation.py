import pandas as pd
import requests
import os
csv_=pd.read_csv("fitzpatrick17k.csv")

disease_list = [
    "psoriasis",
    "allergic contact dermatitis",
    ["lupus erythematosus", "lupus subacute"],
    "folliculitis",
    "scabies",
    "photodermatoses",
    ["acne vulgaris", "acne"],
    ["eczema", "dyshidrotic eczema"],
    "seborrheic dermatitis"]

def cln_data(data, data_list):
    for i in data_list:
        if isinstance(i, list):
            df = data[data["label"].isin(i)]
            disease_name = '_'.join(i)
        else:
            df = data[data["label"] == i]
            disease_name = i

        df.reset_index(inplace=True)
        df["pic_name"] = df["md5hash"]
        data_dict = dict(zip(df["pic_name"], df["url"]))

        folder_path = f"./{disease_name.replace(' ', '_', )}"
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        for pic_name, url in data_dict.items():
            filename = f"{pic_name.replace(' ', '_', )}.jpg"
            file_path = os.path.join(folder_path, filename)

            try:
                headers = {'User-Agent': 'Your Custom User-Agent'}
                response = requests.get(url, headers=headers)

                response.raise_for_status()  # Check for any HTTP 
