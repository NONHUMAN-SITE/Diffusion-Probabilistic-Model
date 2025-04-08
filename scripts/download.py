import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import glob

api = KaggleApi()
api.authenticate()

dataset = 'karnikakapoor/art-portraits'

data_dir = os.path.join(Path(os.path.abspath(__file__)).parent.parent,'data')
os.makedirs(data_dir, exist_ok=True)

#api.dataset_download_files(dataset, path=data_dir, unzip=False)

# Extraer el archivo .zip
file_name = glob.glob(os.path.join(data_dir,"*.zip"))[0]
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

os.remove(file_name)