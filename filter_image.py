from model import *
from triangle_mapping import *
from filter_funcs import *
from filter_video import *

def filter_image(image, filter_name='dog'):
    filters, multi_filter_runtime = load_filter(filter_name)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = DeepFace.extract_faces(frame, (224, 224), detector_backend='opencv')
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
        points2 = landmarks.tolist()
        
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
            
            frame = np.uint8(output)
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return rgb_image

def apply_filter_on_image(image_path, output_file, filter_name='cat'):
    frame = cv2.imread(image_path)
    rgb_image = filter_image(frame, filter_name)
    cv2.imshow("Filter Image", rgb_image)
    cv2.waitKey(0)
    # cv2.imwrite(output_file, rgb_image)
    
if __name__ == '__main__':
    image_path = '../images/bw.jpg'
    apply_filter_on_image(image_path, output_file='./test_images/cat1.jpg', filter_name='anonymous')
    