import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
from deepface import DeepFace
import torchvision.transforms.functional as TF
from timeit import default_timer

image_size = 224

def load_model(model_path='./resnet34.pt'):
    model = torch.load(model_path, map_location='cpu')
    return model.to('cuda:0')

def preprocess_image(image):
    image = TF.to_pil_image(image)
    image = TF.resize(image, (224, 224))
    image = TF.to_tensor(image)
    image = (image - image.min())/(image.max() - image.min()) # Normalize [0, 1]
    image = (2 * image) - 1  # Normalize [-1, 1]
    return image.unsqueeze(0)

def draw_landmarks_on_faces(image, faces_landmarks):
    image = image.copy()
    rad = max(max(image.shape[0], image.shape[1])//200, 1)

    for landmarks, (left, top, height, width) in faces_landmarks:
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5)
        landmarks = landmarks.numpy()
        print(landmarks)

        for i, (x, y) in enumerate(landmarks, 1):
            try:
                cv2.circle(image, (int((x * width) + left), int((y * height) + top)), rad, [255, 255, 0], -1)
            except:
                pass

    return image

@torch.no_grad()
def get_landmarks(frame, model=None):
    if model is None:
        model = load_model()
    frame = np.array(frame)
    landmarks = model(preprocess_image(frame).to('cuda:0')).squeeze()
    landmarks = landmarks.view(-1, 2) + 0.5
    return landmarks.cpu().numpy()
    

@torch.no_grad()
def get_image_landmarks(frame):
    if type(frame) == str:
        frame = np.array(Image.open(frame))
    elif type(frame) == Image.__name__:
        frame = np.array(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = DeepFace.extract_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (224, 224), detector_backend='opencv')

    outputs = []
    model = load_model()

    for i, face in enumerate(faces):
        box = face['facial_area']

        crop_img = gray[box['y']: box['y'] + box['h'], box['x']: box['x'] + box['w']]
        preprocessed_image = preprocess_image(crop_img)
        landmarks_predictions = model(preprocessed_image)
        outputs.append((landmarks_predictions.cpu(), (box['x'], box['y'], box['h'], box['w'])))

    image_landmarks = draw_landmarks_on_faces(frame, outputs)
    plt.imshow(image_landmarks)
    plt.show()
    return image_landmarks
    
def main():
    img_path = './test_images/100032540_1.jpg'
    model = load_model()
    print('Load success')
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    faces = DeepFace.extract_faces(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (224, 224), detector_backend='opencv')
    
    start = default_timer()
    for i, face in enumerate(faces):
        box = face['facial_area']
        crop_img = gray[box['y']: box['y'] + box['h'], box['x']: box['x'] + box['w']]
        landmarks = get_landmarks(crop_img, model)
    end = default_timer()
    print(f'Time: {end-start:.2f}s')  
    for i, (x, y) in enumerate(landmarks, 1):
        try:
            cv2.circle(image, (int(x * box['w'] + box['x']), int(y * box['h'] + box['y'])), 3, [255, 255, 0], -1)
        except:
            pass
    plt.imshow(image)
    plt.show()

    
if __name__ == '__main__':
    main()