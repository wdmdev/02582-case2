# Computational Data Analysis - Case 2
Exploratory (unsupervised) analysis using a cropped and aligned version of the [UTKFaces](https://susanqq.github.io/UTKFace/) dataset.

## Data
The main dataset contains 23705 cropped and aligned face images. The dataset is quite large and therefore not well suited for versioning. Therefore the face image data is ignored through the `.gitignore`, but can be downloaded automatically by running the `make_data.py` script in the folder `src/data`.
<br><br>
Otherwise the data can manually be downloaded from [here](https://files.dtu.dk/userportal/?v=4.5.2#/shared/public/0aKdY0cinWfDop1p/716774b2-c059-4082-8d00-0297b88838d4).
<br>
Then unzip the data and place the contents in the `data` folder. This includes the `Faces` folder, `filenames.txt`, `labels.csv`, and (if you want) `readme.txt` file.

## Repository Structure
The structure of this repo follows the [Cookiecutter Data Science Template](https://drivendata.github.io/cookiecutter-data-science/#directory-structure).

## Authors
August Semrau Andersen, Sunniva Olsrud Punsvik, Søren Winkel Holm, and William Diedrichsen Marstrand.