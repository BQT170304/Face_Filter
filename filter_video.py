import numpy as np
import cv2
from deepface import DeepFace
from model import *
import torch
import time
import math
from filter_funcs import *
import triangle_mapping as tm

def filter_frame(frame: np.array, filter_name: str, model: torch.nn.Module = None, points2Prev = None, img2GrayPrev = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # prepare filter
    filters, multi_filter_runtime = load_filter(filter_name=filter_name)

    # detect faces
    resp_objs = DeepFace.extract_faces(img_path=frame, target_size=(224, 224), detector_backend="opencv", enforce_detection=False)
    if resp_objs is not None:
        for resp_obj in resp_objs:
            # deal with extract_faces
            box = resp_obj["facial_area"]
            if box['w'] == frame.shape[1]:
                break
            cv2.rectangle(frame, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 0, 255), 3)
            input_image = gray[box['y']: box['y'] + box['h'], box['x']: box['x'] + box['w']]
            landmarks = get_landmarks(input_image, model)
            for i, (x, y) in enumerate(landmarks):
                landmarks[i][0], landmarks[i][1] = int(x * box['w'] + box['x']), int(y * box['h'] + box['y'])
                
            landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(box["y"])])])
            landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(box["y"])])])
            points2 = landmarks.tolist()

            ################ Optical Flow and Stabilization Code #####################

            if points2Prev is None and img2GrayPrev is None:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(gray)

            lk_params = dict(winSize=(101, 101), maxLevel=15,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, gray, points2Prev,
                                                            np.array(points2, np.float32),
                                                            **lk_params)

            # Final landmark points are a weighted average of detected landmarks and tracked landmarks

            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = math.exp(-d * d / 50)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
                points2[k] = tm.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = gray
            ################ End of Optical Flow and Stabilization Code ###############
            
            # for point in points2:
            #     cv2.circle(frame, point, 3, (0, 0, 255), -1)
            # applying filter
            for idx, filter in enumerate(filters):

                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter["morph"]:
                    hull1 = filter_runtime['hull']
                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']

                    # create copy of frame
                    warped_img = np.copy(frame)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the delaunay triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        tm.warpTriangle(img1, warped_img, t1, t2)
                        tm.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    tform = tm.similarityTransform(list(points1.values()), dst_points)

                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                
                frame = output = np.uint8(output)
    else:
        print('No face detected')
    return frame, points2Prev, img2GrayPrev

# Apply filter on video/camera
def apply_filter_on_video(source, filter_name = "squid_game_front_man", output_path = None)->None:
    # prepare video capture
    cap = cv2.VideoCapture(source)

    # create the output video file
    if source != 0 and output_path != None:
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 20
        print('FPS:', fps)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path + 'lm_video.mp4', fourcc, fps, frame_size)

    # create model
    model = load_model()

    # prepare for filter
    if filter_name is None:
        iter_filter_keys = iter(filters_config.keys())
        filter_name = next(iter_filter_keys)
    
    # for optical flow
    points2Prev = None
    img2GrayPrev = None

    count = 0
    prev_time = 0
    # loop through each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if source == 0:
            frame = cv2.flip(frame, 1)

        # apply filter
        frame, points2Prev, img2GrayPrev = filter_frame(frame, filter_name, model, points2Prev, img2GrayPrev)

        if source == 0:
            # fps
            # cur_time = time.time()
            # print(1 / (cur_time - prev_time))
            # prev_time = cur_time

            # show frame
            cv2.imshow("Filter app", frame)

            # handle keypress
            keypressed = cv2.waitKey(1) & 0xFF
            if keypressed == ord('q'):
                break
            elif keypressed == ord('f'):
                try:
                    filter_name = next(iter_filter_keys)
                except:
                    iter_filter_keys = iter(filters_config.keys())
                    filter_name = next(iter_filter_keys)
        else:
            if output_path != None:
                out.write(frame) 
                count += 1 
                print(count)
    
    # save & free resource
    cap.release()
    if source != 0 and output_path != None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    apply_filter_on_video(0, filter_name='cat', output_path='./')
    