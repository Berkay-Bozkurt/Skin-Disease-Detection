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


