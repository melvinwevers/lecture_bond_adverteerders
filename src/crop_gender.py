import glob
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import argparse
from utils import find_bb, to_parseable


'''
Script to crop faces from images to generate training data for gender detection
algorithm
'''

parser = argparse.ArgumentParser(description='Create Gender Training Data')
parser.add_argument('--annotation_dir',
                    default='../../advertisinggender/data/raw/annotations/',
                    type=str, help='dir that holds annotations')
parser.add_argument('--images_dir',
                    default='../../advertisinggender/data/processed/KB_FACES/',
                    type=str, help='dir with images')
parser.add_argument('--output_dir',
                    default='../data/processed/faces/',
                    type=str, help='output for cropped images')

args = parser.parse_args()


def crop_gender():
    annotation_dir = args.annotation_dir
    images_dir = args.images_dir

    annotations = glob.glob(annotation_dir + '*')

    for index, file in enumerate(annotations):
        print(file)
        print("processing: {}".format(index))

        male_counter = 0
        female_counter = 0
        for xml in tqdm(annotations[index]):
            try:
                tree = ET.parse(xml).getroot()
                tree = to_parseable(tree)

                folder = tree.find("folder").text
                filename = tree.find("filename").text
                file_path = images_dir + '{}/{}'.format(folder, filename)
                img = cv2.imread(file_path)
                print(img)
                padding = 25
                objects = tree.findall('.//object')
                for object in objects:
                    if object.find("name").text in ['f']:
                        xmin, ymin, xmax, ymax = find_bb(object)

                        if (ymax - ymin) >= 100 and (xmax - xmin) >= 100:


                        # TODO: check if this is necessary
                        # if ymin > 25:
                        #     ymin -= 25
                        # if xmin > 25:
                        #     xmin -= 25
                            crop = np.copy(img[ymin:(ymax + padding),
                                               xmin:(xmax + padding)])
                            if 0 not in crop.shape[:2]:  # check if crop != empty
                                cv2.imwrite(args.output_dir + '/f/{}_{}.jpg'.format(folder,
                                                                                    female_counter), crop)
                                female_counter += 1
                            else:
                                pass
                        else:
                            pass

                    elif object.find("name").text in ['m']:
                        xmin, ymin, xmax, ymax = find_bb(object)
                        if (ymax - ymin) >= 100 and (xmax - xmin) >= 100:

                            crop = np.copy(img[(ymin - padding):(ymax + padding),
                                               (xmin - padding):(xmax + padding)])
                            if 0 not in crop.shape[:2]:
                                cv2.imwrite(
                                    args.output_dir + '/m/{}_{}.jpg'.format(folder, male_counter), crop)
                                male_counter += 1
                            else:
                                pass
                        else:
                            pass
            except Exception:
                pass


if __name__ == '__main__':
    crop_gender()
