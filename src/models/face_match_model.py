import os
import sys
from PIL import Image
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from src.features.load_image import load_img_as_gray

import pickle as pkl
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--model', '-m', type=str, required=True, 
                        help='Name of model to load, which must be placed inside data/models')
    parser.add_argument('--image_folder', '-if', type=str, required=True, 
                        help='Name of image folder, which must be placed inside data/prediction')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder name e.g. profile_pic_matches')

    args = parser.parse_args()

    # Remove file type ending if someone puts it in the model argument
    args.model = os.path.splitext(args.model)[0]

    model_base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models')
    image_folder_base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data', 'face_match')
    matches_output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'reports', args.output)

    if not os.path.exists(matches_output_path):
        os.mkdir(matches_output_path)

    # Load model
    with open(os.path.join(model_base_path, args.model+'.pkl'), 'rb') as model_file:
        model = pkl.load(model_file)

    # Find face matches
    print(f'Finding best matches for images in: {image_folder_base_path}')
    for img_file in tqdm(sorted(os.listdir(os.path.join(image_folder_base_path, args.image_folder)))):
        img_path = os.path.join(image_folder_base_path, args.image_folder, img_file)
        img = load_img_as_gray(img_path)
        # Best face as numpy array
        best_face_array, id = model.find_best_face_match(img)

        img_name = os.path.splitext(img_file)[0]
        img_match_path = os.path.join(matches_output_path, f'img_{img_name}_best_match_img{id}.jpg')
        best_face_img = Image.fromarray(best_face_array)
        best_face_img.save(img_match_path)