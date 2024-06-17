import torch
import cv2
from deepface import DeepFace
from ultralytics import YOLO
from model import *

if __name__ == '__main__':
    # yolo = YOLO('./yolo.pt')
    model = torch.load('./resnet34.pt')
    image_path = './test_images/100040721_2.jpg'
    # res = yolo(image_path)
    # xywh = res[0].boxes[0].xywh.cpu().numpy()[0]
    # xyxy = res[0].boxes[0].xyxy.cpu().numpy()[0]
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # x,y,w,h = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])
    # x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

    
    # crop_img = gray[y1:y2, x1:x2]
    # cv2.imwrite('./yolo_bqt.jpg', cv2.resize(crop_img, (224, 224)))
    
    # landmarks = get_landmarks(crop_img, model)
    # for i, (xi, yi) in enumerate(landmarks):
    #     landmarks[i][0], landmarks[i][1] = int(xi * w + x), int(yi * h + y)
    # points = landmarks.tolist()
    # print(x, y, w, h)
    # print(landmarks)
    # for point in points:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    
    resp_objs = DeepFace.extract_faces(img_path=img, target_size=(224, 224), detector_backend="opencv", enforce_detection=False)
    if resp_objs is not None:
        for resp_obj in resp_objs:
            # deal with extract_faces
            box = resp_obj["facial_area"]
            if box['w'] == img.shape[1]:
                break
            # cv2.rectangle(frame, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 0, 255), 3)
            input_image = gray[box['y']: box['y'] + box['h'], box['x']: box['x'] + box['w']]
            
            landmarks_list = []
            for _ in range(5):
                temp = get_landmarks(input_image, model)
                landmarks_list.append(temp)
            landmarks = np.mean(np.array(landmarks_list), axis=0)
            print(landmarks.shape)
            for i, (x, y) in enumerate(landmarks):
                landmarks[i][0], landmarks[i][1] = int(x * box['w'] + box['x']), int(y * box['h'] + box['y'])
            # print(landmarks2)
            for point in landmarks:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
    
    cv2.imshow("Deepface", img)
    cv2.waitKey(0)
    # plt.imshow(img)
    # plt.show()

    
