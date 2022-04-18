import os
import shutil
from tqdm import tqdm
import requests as req
from zipfile import ZipFile 

FACES_FILE_URI1 = 'https://files.dtu.dk/fss/public/link/public/stream/read/data.zip?linkToken=0aKdY0cinWfDop1p&itemName=716774b2-c059-4082-8d00-0297b88838d4'
ZIP_FILE_NAME = 'faces.zip'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE_PATH = os.path.join(BASE_PATH, '..', '..', 'data', 'raw', ZIP_FILE_NAME)
EXTRACT_PATH = os.path.join(BASE_PATH, '..', '..', 'data', 'raw')


def get_data():
    print('Downloading face data...')
    r = req.get(FACES_FILE_URI1, stream=True)

    print(f'Saving face data to {ZIP_FILE_PATH}')
    total_size_in_bytes= int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(ZIP_FILE_PATH, 'wb') as faces:
        for data in r.iter_content(block_size):
            progress_bar.update(len(data))
            faces.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print('Unzipping data...')
    with ZipFile(ZIP_FILE_PATH, 'r') as zip:
        zip.extractall(EXTRACT_PATH)


    for name in os.listdir(os.path.join(EXTRACT_PATH, 'data')):
        shutil.move(os.path.join(EXTRACT_PATH, 'data', name), os.path.join(EXTRACT_PATH, name))

    os.removedirs(os.path.join(EXTRACT_PATH, 'data'))
    os.remove(ZIP_FILE_PATH)

    print('Data created sucessfully!')