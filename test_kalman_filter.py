from model import *
from triangle_mapping import *
from filter_funcs import *
from filter_video import *
from kalman_filter import KalmanFilter
import pathlib

# def find_convex_hull(points):
#     hull = []
#     hullIndex = cv2.convexHull(np.array(points), clockwise=False, returnPoints=False)
#     addPoints = [
#         [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
#         [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
#         [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
#         [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
#         [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
#     ]
#     hullIndex = np.concatenate((hullIndex, addPoints))
#     for i in range(0, len(hullIndex)):
#         hull.append(points[hullIndex[i][0]])

#     return hull, hullIndex

# # Draw a point
# def draw_point(img, p, color) :
#     cv2.circle( img, (int(p[0]), int(p[1])), 2, color, -1)
 
# # Draw delaunay triangles
# def draw_delaunay(img, subdiv, delaunay_color ) :
 
#     triangleList = subdiv.getTriangleList();
#     size = img.shape
#     r = (0, 0, size[1], size[0])
 
#     for t in triangleList :
 
#         pt1 = (int(t[0]), int(t[1]))
#         pt2 = (int(t[2]), int(t[3]))
#         pt3 = (int(t[4]), int(t[5]))
 
#         if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3) :
 
#             cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
#             cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
#             cv2.line(img, pt3, pt1, delaunay_color, 1, 0)
            
# Initialize kalman filters
kalman_filters = [KalmanFilter(dt=1/20, u_x=0.5, u_y=0.5, std_acc=1, x_std_meas=0.1, y_std_meas=0.1) for _ in range(70)]

def filter_frame_with_kmf(frame: np.array, filter_name: str, model: torch.nn.Module = None, points2Prev = None, img2GrayPrev = None):
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
            landmarks_list = []
            for _ in range(5):
                temp = get_landmarks(input_image, model)
                landmarks_list.append(temp)
            landmarks = np.mean(np.array(landmarks_list), axis=0)
            for i, (x, y) in enumerate(landmarks):
                landmarks[i][0], landmarks[i][1] = int(x * box['w'] + box['x']), int(y * box['h'] + box['y'])
                
            landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(box["y"])])])
            landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(box["y"])])])
            
            for i, point in enumerate(landmarks):
                kalman_filter = kalman_filters[i]

                # Prediction step
                kalman_filter.predict()

                # Update step
                kalman_filter.update(point.reshape(2, 1))

                # Get the estimated state
                estimated_state = kalman_filter.x[:2]

                # Assign to landmarks
                landmarks[i][0], landmarks[i][1] = int(estimated_state[0]), int(estimated_state[1])
                
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
def apply_filter_on_video_kmf(source, filter_name = "squid_game_front_man", output_path = None)->None:
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
        frame, points2Prev, img2GrayPrev = filter_frame_with_kmf(frame, filter_name, model, points2Prev, img2GrayPrev)

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

def main():
    cap = cv2.VideoCapture(0)
    # create model
    model = load_model()
    
    count = 0
    # loop through each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        count += 1
        # Draw landmarks on frame
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = DeepFace.extract_faces(frame, (224, 224), detector_backend='opencv', enforce_detection=False)
        if faces is None:
            return
        for face in faces:
            box = face['facial_area']
            crop_img = gray[box['y']: box['y'] + box['h'], box['x']: box['x'] + box['w']]
            landmarks = get_landmarks(crop_img)
            for i, (x, y) in enumerate(landmarks):
                landmarks[i][0], landmarks[i][1] = int(x * box['w'] + box['x']), int(y * box['h'] + box['y'])
            landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(box["y"])])])
            landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(box["y"])])])
            # points = landmarks.tolist()
            for i, point in enumerate(landmarks):
                kalman_filter = kalman_filters[i]

                # Prediction step
                kalman_filter.predict()

                # Update step
                kalman_filter.update(point.reshape(2, 1))

                # Get the estimated state
                estimated_state = kalman_filter.x[:2]

                # Draw the estimated state on the frame
                cv2.circle(frame, (int(estimated_state[0]), int(estimated_state[1])), 3, (0, 255, 0), -1)
        
        cv2.imshow("Kalman Filter on video", frame)

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

    # save & free resource
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    apply_filter_on_video_kmf(0, filter_name='dog', output_path='./')
    # main()