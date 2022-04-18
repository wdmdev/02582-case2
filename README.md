# Computational Data Analysis - Case 2
Exploratory (unsupervised) analysis using a cropped and aligned version of the [UTKFaces](https://susanqq.github.io/UTKFace/) dataset.

## Data
The main dataset contains 23705 cropped and aligned face images. The dataset is quite large and therefore not well suited for versioning. Therefore the face image data is ignored through the `.gitignore`, but can be downloaded automatically by running the `make_data.py` script in the folder `src/data`.
<br><br>
Otherwise the data can manually be downloaded from [here](https://files.dtu.dk/userportal/?v=4.5.2#/shared/public/0aKdY0cinWfDop1p/716774b2-c059-4082-8d00-0297b88838d4).
<br>
Then unzip the data and place the contents in the `data/raw` folder. This includes the `Faces` folder, `filenames.txt`, `labels.csv`, and (if you want) `readme.txt` file.

## Repository Structure
The structure of this repo follows the [Cookiecutter Data Science Template](https://drivendata.github.io/cookiecutter-data-science/#directory-structure).

### data
All raw data should be saved to the `data/raw` folder. When data is preprocessed through code it should be saved (if necessary) to the `data/preprocessed` folder.

### models
If models are saved after training for later use, they should be saved to the `models` folder.

### notebooks
All notebooks should be saved to the `notebooks` folder.

### reports
If analysis are saved, they should be saved to the `reports` folder and specifically figures/plots needs to be saved to the `reports/figures` folder.

### src
The `src` folder should contain **all** code for the project except that in the `notebooks`.

#### src/data
The `src/data` folder should contain code that generates or downloads data to the `data/raw` folder.

#### src/features
The `src/features` folder should contain code that generates data features from the raw data in `data/raw` and saves it to the `data/preprocessed` folder.

#### src/models
The `src/models` folder should contain code for training a model and saving it to `models` in the root folder if necessary or loading a model from `models` and perform predictions.

#### src/visualization
The `src/visualization` folder should contain code for generating figures/plots and saving them to `reports/figures`.

## Authors
August Semrau Andersen, Sunniva Olsrud Punsvik, Søren Winkel Holm, and William Diedrichsen Marstrand.