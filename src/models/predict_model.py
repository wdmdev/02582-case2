import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from argparse import ArgumentParser
import pickle as pkl
from tqdm import tqdm
import pandas as pd

from src.features.load_image import load_img_as_gray

if __name__ == '__main__':
    arg_parser = ArgumentParser(description='Use --model to predict (age, race, gender) for all images in --image_folder')

    arg_parser.add_argument('--model', '-m', type=str, required=True, 
                            help='Name of model to load, which must be placed inside data/models')
    arg_parser.add_argument('--image_folder', '-if', type=str, required=True, 
                            help='Name of image folder, which must be placed inside data/prediction')
    arg_parser.add_argument('--output', '-o', type=str, required=True,
                            help='Output file name (without file type) e.g. my_profile_pic_predictions')

    args = arg_parser.parse_args()

    # Remove file type ending if someone puts it in the model or output argument
    args.model = os.path.splitext(args.model)[0]
    args.output = os.path.splitext(args.output)[0]

    model_base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models')
    image_folder_base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data', 'prediction')
    prediction_output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'reports', args.output+'.csv')

    # Load model
    with open(os.path.join(model_base_path, args.model+'.pkl'), 'rb') as model_file:
        model = pkl.load(model_file)

    # Make predictions for each image in the arg.image_folder and save the predictions to the reports folder
    print('Making predictions...')
    prediction_dict = {'image': [], 'age': [], 'race': [], 'gender': []}
    for img_file in tqdm(sorted(os.listdir(os.path.join(image_folder_base_path, args.image_folder)))):
        img_path = os.path.join(image_folder_base_path, args.image_folder, img_file)
        img = load_img_as_gray(img_path)
        age, race, gender = model.predict(img.flatten().reshape(1,-1))

        img_name = os.path.splitext(img_file)[0]
        prediction_dict['image'].append(img_name)
        prediction_dict['age'].extend(age)
        prediction_dict['race'].extend(race)
        prediction_dict['gender'].extend(gender)

    prediction_df = pd.DataFrame(prediction_dict)
    prediction_df.to_csv(prediction_output_path, index=False)
    print(f'Saved predictions to {prediction_output_path}')