from pathlib import Path
import xml.etree.ElementTree as ET
import os
import cv2
import io
from PIL import Image
import pickle
from torchvision import transforms, models
from torch.autograd import Variable
import random
import numpy as np
import torch
from scipy.spatial import distance as dist


datapath = '../data/processed/KB_FACES'

vgg16 = models.vgg16(pretrained=True)
newmodel = torch.nn.Sequential(*(list(vgg16.features[:24])))


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def to_parseable(tree):
    t = ET.tostring(tree)
    t = t.lower()
    return ET.fromstring(t)


def calculate_mean(files_, sample_size=1000, means=[]):
    '''
    take a sample of the list of images and calculate the mean R,G,B value
    '''
    sample_list = random.sample(files_, sample_size)
    for file_ in sample_list:
        img = cv2.imread(file_)
        means.append((cv2.mean(img)))
    print(np.mean(means, axis=0))
    print(np.std(means, axis=0))


def find_bb(object):
    xmax = int(object.find('.//xmax').text)
    xmin = int(object.find('.//xmin').text)
    ymax = int(object.find('.//ymax').text)
    ymin = int(object.find('.//ymin').text)
    return xmin, ymin, xmax, ymax


def detect_dnn(file_, net):
    # TO DO Change averages use default or based on dataset???
    '''
    detecting faces using openCV's deep neural network face detector
    The output is a dictionary with confidence scores and x,y,w,h
    '''
    img = cv2.imread(file_)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (150.23, 170.07, 181.21))
    #                             (300, 300), (104, 177, 123))

    net.setInput(blob)
    detections = net.forward()
    pred_box = {}
    pred_box['boxes'] = []
    pred_box['scores'] = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0:
            locations = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # (startX, startY, endX, endY) = locations.astype("int")
            pred_box['boxes'].append([int(locations[0]),
                                      int(locations[1]),
                                      int(locations[2]),
                                      int(locations[3])
                                      ])
            pred_box['scores'].append(float('{:.3f}'.format(confidence)))
    return pred_box


def generate_dt(file_, object_, file_base, DT_PATH, net):
    '''
    function to detect faces in annotated images using open cv's dnn
    '''
    pred_boxes = {}
    pred_box = detect_dnn(file_, net)
    pred_boxes[file_base] = pred_box
    os.makedirs(DT_PATH, exist_ok=True)
    with open(DT_PATH+'/{}.txt'.format(file_base), 'w') as f:
        scores = np.array(pred_box['scores']).tolist()
        boxes = np.array(pred_box['boxes']).tolist()
        if not scores:
            f.write(str(object_)+" "+str(0)+" "+str(0) +
                    " "+str(0)+" "+str(0)+" "+str(0))
        else:
            for box, score in zip(boxes, scores):
                f.write(str(object_) + " "+str(float('{:.3f}'.format(score)))+" "+str(
                    box[0])+" "+str(box[1])+" "+str(box[2])+" "+str(box[3])+"\n")


def generate_gt(xml, object_, file_base, GT_PATH):
    '''
    function to extract gt (ground-truth) files from annotations xml
    '''
    gt_boxes = {}
    gt_box = get_annotations(xml)
    if not any(gt_box):  # check if there are face annotations
        pass
    else:
        gt_boxes[file_base] = gt_box
        os.makedirs(GT_PATH, exist_ok=True)
        with open(GT_PATH+'/{}.txt'.format(file_base), 'w') as f:
            for gt in gt_box:
                # left_xmin, top_ymin, right_xmax, bottom_ymax
                f.write(
                    str(object_)+" "+str(gt[1])+" "+str(gt[0])+" "+str(gt[3])+" "+str(gt[2])+"\n")


def generate_gt_WIDER(xml, object_, file_base, GT_PATH):
    '''
    function to extract gt (ground-truth) files from annotations xml
    '''
    gt_boxes = {}
    gt_box = get_annotations(xml)
    if not any(gt_box):  # check if there are face annotations
        pass
    else:
        gt_boxes[file_base] = gt_box
        os.makedirs(GT_PATH, exist_ok=True)
        with open(GT_PATH+'/{}.txt'.format(file_base), 'w') as f:
            for gt in gt_box:
                # left, top, width, height 
                left = gt[0]
                top = gt[1]
                width = gt[2] - gt[0]
                height = gt[4] - gt[1]

                f.write(
                    str(left)+" "+str(gt[0])+" "+str(gt[3])+" "+str(gt[2])+"\n")


def get_annotations(xml):
    img_data = []
    tree = ET.parse(xml).getroot()
    tree = to_parseable(tree)
    objects = tree.findall('.//object')
    # folder = tree.find("folder").text

    for object in objects:
        if object.find("name").text in ['m', 'b', 'g', 'woman', 'man', 'f']:
            class_name = object.find("name").text
            if class_name == 'woman' or class_name == 'g':
                class_name = 'f'
            elif class_name == 'man' or class_name == 'b':
                class_name = 'm'
            x1, x2, y1, y2 = find_bb(object)
            img_data.append([int(x1), int(y1),
                             int(x2), int(y2)])
            # str(class_name)])

    return img_data


preprocess_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


def classify_single_image(img, newmodel=newmodel):
    filename = "../notebooks/finalized_model.sav"
    clf = pickle.load(open(filename, 'rb'))
    img_tensor = preprocess_img(img)
    img_tensor.unsqueeze_(0)
    images = Variable(img_tensor)
    encoding = newmodel(images)
    encoding = encoding.detach().numpy().flatten()
    prediction = clf.predict((np.asarray(encoding).reshape(1, -1)))
    return prediction


def enrich_annotations(xml):
    img_data = []
    tree = ET.parse(xml).getroot()
    tree = to_parseable(tree)
    objects = tree.findall('.//object')
    filename = tree.find("filename").text
    folder = tree.find("folder").text

    for object in objects:
        if object.find("name").text in ['m', 'b', 'g', 'woman', 'man', 'f']:
            class_name = object.find("name").text
            if class_name == 'woman' or class_name == 'g':
                class_name = 'f'
            elif class_name == 'man' or class_name == 'b':
                class_name = 'm'
            x1, x2, y1, y2 = find_bb(object)
            img_data.append([int(x1), int(y1),
                             int(x2), int(y2)])
            img_path = datapath + '/' + folder + '/' + filename
            img = Image.open(img_path)
            img_crop = img.crop((x1, y1, x2, y2))
            prediction = classify_single_image(img_crop)
            print(prediction)

            # str(class_name)])

    return img_data


# TO DO: MOVE THIS TO FUNCTION for other project

def get_men_women_annotations(xml, prediction=False):
    '''
    This function extracts the males and females from the annotations. 
    We also calculate the relative area of these annotated faces, and the 
    distance of males and females from the center of the image. 
    TODO: CLEAN THIS UP AS A CLASS
    '''
    men = 0
    areas_m = []
    areas_f = []
    distances_m = []
    type_m = []
    type_f = []
    distances_f = []
    object_m = []
    object_f = []
    position_m = []
    position_f = []
    women = 0
    tree = ET.parse(xml).getroot()
    tree = to_parseable(tree)
    filename = tree.find("filename").text
    folder = tree.find("folder").text
    height = int(tree.findall('.//height')[0].text)
    width = int(tree.findall('.//width')[0].text)
    total_area = height * width
    objects = tree.findall('.//object')
    img_path = datapath + '/' + folder + '/' + filename
    for object_ in objects:
        if object_.find("name").text in ['m', 'man', 'b']:
            xmin, ymin, xmax, ymax = find_bb(object_)   
            area = (ymax-ymin) * (xmax-xmin)
            img = Image.open(img_path)
            img_crop = img.crop((xmin, ymin, xmax, ymax))
            if prediction:
                pred = classify_single_image(img_crop)
                type_m.append(int(prediction))
            rel_area = area/total_area
            areas_m.append(rel_area)
            D, position, object_center = distance_from_center(width, height, object_)
            position_m.append(position)
            distances_m.append(D)
            object_m.append(object_center)
            men += 1
        if object_.find("name").text in ['f', 'woman', 'g']:
            xmin, ymin, xmax, ymax = find_bb(object_)
            area = (ymax-ymin) * (xmax-xmin)
            img = Image.open(img_path)
            img_crop = img.crop((xmin, ymin, xmax, ymax))
            if prediction:
                pred = classify_single_image(img_crop)
                type_f.append(int(prediction))
            rel_area = area/total_area
            areas_f.append(rel_area)
            D, position, object_center = distance_from_center(width, height, object_)
            position_f.append(position)
            distances_f.append(D)
            object_f.append(object_center)
            women += 1
    return men, women, areas_m, areas_f, total_area

def distance_from_center(width, height, object_):
    ymin, ymax, xmin, xmax = find_bb(object_)
    image_center = ((height / 2), (width / 2))
    #object_height = ymax-ymin
    #object_width = xmax-xmin
    object_center = (((xmax+xmin)/2)/width), ((ymax+ymin)/2/height)
    #print(object_center)
    if object_center[0] > 0.5 and object_center[1] > 0.5:
        position = 'UR'
    elif object_center[0] > 0.5 and object_center[1] <= 0.5:
        position = 'LR'
    elif object_center[0] <= 0.5 and object_center[1] <= 0.5:
        position = 'LL'
    else:
        position = 'UL'
    D = dist.euclidean((0.5, 0.5), object_center)
    #rel_position = np.subtract(object_center, (0.5, 0.5))


    return D, position, object_center


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0,
                      v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img
    return eraser
