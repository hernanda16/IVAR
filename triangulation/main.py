import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# X AXIS IS BASED OFF THE LONGEST SIDE OF THE FIELD
# ALL WORLD LENGTHS ARE IN CM
FIELD_X = 1200
FIELD_Y = 800

IMAGE_0_150_YUV = [(121, 78, 192), (255, 100, 255)]
IMAGE_0_650_YUV = [(0, 0, 202), (255, 255, 255)]
IMAGE_600_0_YUV = [(0, 0, 157), (85, 117, 219)]
IMAGE_600_800_YUV = [(0, 0, 194), (255, 255, 255)]

# Camera actual position in cm
CAM_FIELD_0_150 = [0, 150]
CAM_FIELD_0_650 = [0, FIELD_Y - 150]
CAM_FIELD_600_0 = [FIELD_X / 2, 0]
CAM_FIELD_600_800 = [FIELD_X / 2, FIELD_Y]

# Where the center of the field is relative to cameras pixel
CAM_600_0_MID_FIELD = [2005, 1733]
CAM_600_800_MID_FIELD = [2005, 1733]

def cm2px(x, y):
    x = int(x + 100)
    y = int(y + 100)
    return x, y

def calibrateCamera(image_path, checkerboard_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    image_glob = glob.glob(image_path)

    for fname in image_glob:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs

def findEpiPolar(image_1, image_2):
    image_1 = cv2.cvtColor(image_1, cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.cvtColor(image_2, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    points_1 = []
    points_2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            points_1.append(keypoints_1[m.queryIdx].pt)
            points_2.append(keypoints_2[m.trainIdx].pt)

    points_1 = np.int32(points_1)
    points_2 = np.int32(points_2)
    
    F, mask = cv2.findFundamentalMat(points_1, points_2, cv2.FM_RANSAC)
    points_1 = points_1[mask.ravel() == 1]
    points_2 = points_2[mask.ravel() == 1]
    
    return points_1, points_2

def rectifyImages(image_1, image_2, mtx, dist, points_1, points_2):
    E, mask = cv2.findEssentialMat(points_1, points_2, mtx)
    points_1 = points_1[mask.ravel() == 1]
    points_2 = points_2[mask.ravel() == 1]
    h, w = image_1.shape[:2]

    _, R, T, _ = cv2.recoverPose(E, points_1, points_2, mtx)
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx, dist, mtx, dist, (w, h), R, T)

    map1_x, map1_y = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (w, h), cv2.CV_32FC1)

    map_width = max(map1_x.shape[1], map2_x.shape[1])
    map_height = max(map1_x.shape[0], map2_x.shape[0])

    map1_x = cv2.resize(map1_x, (map_width, map_height), interpolation=cv2.INTER_LINEAR)
    map1_y = cv2.resize(map1_y, (map_width, map_height), interpolation=cv2.INTER_LINEAR)
    map2_x = cv2.resize(map2_x, (map_width, map_height), interpolation=cv2.INTER_LINEAR)
    map2_y = cv2.resize(map2_y, (map_width, map_height), interpolation=cv2.INTER_LINEAR)

    rectified_image_1 = cv2.remap(image_1, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_image_2 = cv2.remap(image_2, map2_x, map2_y, cv2.INTER_LINEAR)

    return rectified_image_1, rectified_image_2

def detectBall(image, yuv_range):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    mask_1 = cv2.inRange(image, yuv_range[0], yuv_range[1])
    contours, _ = cv2.findContours(mask_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 200:
            (x, y), _ = cv2.minEnclosingCircle(contours[i])
            ball_center = (int(x), int(y))

    return ball_center

def computeDistanceFromBaseline(points_1, points_2, cam_pos_1, cam_pos_2, camera_height, camera_matrix=None):
    points_1 = np.array(points_1)
    points_2 = np.array(points_2)

    focal_length_x = camera_matrix[0][0]

    disparity = abs(points_1[0] - points_2[0])
    distance_between_cameras = np.sqrt((cam_pos_1[0] - cam_pos_2[0])**2 + (cam_pos_1[1] - cam_pos_2[1])**2)

    depth = (distance_between_cameras * focal_length_x) / disparity
    
    distance_from_baseline = np.sqrt(depth**2 - camera_height**2)
    return distance_from_baseline

def undistortImage(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

def computeDistanceFromMiddle(image, cam_pos, ball_points, middlefield_points, direction=1):
    # Still uses linear regression
    px2cm =  (FIELD_Y / 2) / abs(image.shape[0] - middlefield_points[1])
    distance = abs(image.shape[0] - ball_points[1]) * px2cm

    if direction > 0:
        return distance + cam_pos[1]
    else:
        return cam_pos[1] - distance
    
def drawPointsOnImage(image, point):
    cv2.circle(image, (int(point[0]), int(point[1])), 50, (0, 0, 255), -1)
    return image

def detectField(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_channel, s_channel, v_channel = cv2.split(image_hsv)
    h_channel_hist = cv2.calcHist([h_channel], [0], None, [256], [0, 256])
    h_channel_hist = h_channel_hist.flatten()

    h_channel_hist_avg = np.average(h_channel_hist)

    h_channel_hist_filtered = [value if value >= h_channel_hist_avg/2.0 else 0 for value in h_channel_hist]

    h_channel_hist_filtered_max_index = np.argmax(h_channel_hist_filtered)
    for i in range(len(h_channel_hist_filtered)):
        if h_channel_hist_filtered[h_channel_hist_filtered_max_index - i] == 0:
            break
        h_channel_hist_filtered_min_range = h_channel_hist_filtered_max_index - i
    for i in range(len(h_channel_hist_filtered)):
        if h_channel_hist_filtered[h_channel_hist_filtered_max_index + i] == 0:
            break
        h_channel_hist_filtered_max_range = h_channel_hist_filtered_max_index + i
    
    mask = cv2.inRange(image_hsv, np.array([h_channel_hist_filtered_min_range, 0, 0]), np.array([h_channel_hist_filtered_max_range, 255, 255]))
    
    # Extract S and V channel values from pixels where mask is non-zero
    s_channel_range = s_channel[mask > 0]
    v_channel_range = v_channel[mask > 0]

    # Store original min/max for S channel before any filtering
    s_channel_min_original = np.min(s_channel_range) if s_channel_range.size > 0 else 0
    s_channel_max_original = np.max(s_channel_range) if s_channel_range.size > 0 else 255
    
    # Apply additional filtering criteria if needed
    # For example, you could filter based on intensity or other properties
    # This example filters S channel values that are above the mean
    s_mean = np.mean(s_channel_range) if s_channel_range.size > 0 else 0
    filtered_s_channel = s_channel_range[s_channel_range > s_mean] if s_channel_range.size > 0 else np.array([])
    
    # Get min/max of filtered S channel values
    s_channel_min = np.min(filtered_s_channel) if filtered_s_channel.size > 0 else s_channel_min_original
    s_channel_max = np.max(filtered_s_channel) if filtered_s_channel.size > 0 else s_channel_max_original
    
    # Get V channel min/max
    v_channel_min = np.min(v_channel_range) if v_channel_range.size > 0 else 0
    v_channel_max = np.max(v_channel_range) if v_channel_range.size > 0 else 255

    image_hsv = cv2.inRange(image_hsv, np.array([h_channel_hist_filtered_min_range, s_channel_min, v_channel_min]), np.array([h_channel_hist_filtered_max_range, s_channel_max, v_channel_max]))
    
    image = cv2.bitwise_and(image, image, mask=image_hsv)

    return image

def drawLineFromBaseline(image_field, cam_points_1, cam_points_2, distance_from_baseline, direction=1, vertical=True, from_middle=False):
    cam_points_1 = cm2px(cam_points_1[0], cam_points_1[1])
    cam_points_2 = cm2px(cam_points_2[0], cam_points_2[1])

    baseline_mid_point = ((cam_points_1[0] + cam_points_2[0]) / 2, (cam_points_1[1] + cam_points_2[1]) / 2)

    angle = np.arctan2(cam_points_2[1] - cam_points_1[1], cam_points_2[0] - cam_points_1[0])

    if direction > 0:
        angle -= np.pi / 2
    else:
        angle += np.pi / 2
    
    end_point_x = int(distance_from_baseline * np.cos(angle) + baseline_mid_point[0])
    end_point_y = int(distance_from_baseline * np.sin(angle) + baseline_mid_point[1])
    
    # Distance Line Init
    if vertical:
        long_line_x0 = end_point_x
        long_line_y0 = 0
        long_line_x1 = end_point_x
        long_line_y1 = image_field.shape[0]
        
        long_line_x0 = int(long_line_x0 * np.cos(angle) - long_line_y0 * np.sin(angle))
        # long_line_y0 = int(long_line_x0 * np.sin(angle) + long_line_y0 * np.cos(angle))
        long_line_x1 = int(long_line_x1 * np.cos(angle) - long_line_y1 * np.sin(angle))
        # long_line_y1 = int(long_line_x1 * np.sin(angle) + long_line_y1 * np.cos(angle))
    else:
        long_line_x0 = 0
        long_line_y0 = end_point_y
        long_line_x1 = image_field.shape[1]
        long_line_y1 = end_point_y

        long_line_y0 = int(long_line_x0 * np.cos(angle) - long_line_y0 * np.sin(angle))
        # long_line_y0 = int(long_line_x0 * np.sin(angle) + long_line_y0 * np.cos(angle))
        long_line_y1 = int(long_line_x1 * np.cos(angle) - long_line_y1 * np.sin(angle))
        # long_line_y1 = int(long_line_x1 * np.sin(angle) + long_line_y1 * np.cos(angle))

    if from_middle:
        long_line_x0 = abs(long_line_x0)
        long_line_y0 = abs(long_line_y0)
        long_line_x1 = abs(long_line_x1)
        long_line_y1 = abs(long_line_y1)

    print(f"long_line_x0: {long_line_x0}, long_line_y0: {long_line_y0}")
    print(f"long_line_x1: {long_line_x1}, long_line_y1: {long_line_y1}")
    print(f"verical: {vertical}, direction: {direction}")

    cv2.line(image_field, (int(cam_points_1[0]), int(cam_points_1[1])), (int(cam_points_2[0]), int(cam_points_2[1])), (255, 0, 0), 2)
    cv2.line(image_field, (int(long_line_x0), int(long_line_y0)), (int(long_line_x1), int(long_line_y1)), (0, 0, 255), 2)
    cv2.line(image_field, (int(baseline_mid_point[0]), int(baseline_mid_point[1])), (end_point_x, end_point_y), (255, 0, 255), 2)

    cv2.circle(image_field, (int(cam_points_1[0]), int(cam_points_1[1])), 20, (255, 0, 0), -1)
    cv2.circle(image_field, (int(cam_points_2[0]), int(cam_points_2[1])), 20, (255, 0, 0), -1)

    return image_field

def drawField():
    field = np.zeros((1000, 1400, 3), dtype=np.uint8)
    field[:] = (0, 128, 0) 

    cv2.rectangle(field, cm2px(0, 0), cm2px(FIELD_X, FIELD_Y), (255, 255, 255), 2)

    cv2.line(field, cm2px(FIELD_X / 2, 0), cm2px(FIELD_X / 2, FIELD_Y), (255, 255, 255), 2)
    cv2.circle(field, cm2px(FIELD_X / 2, FIELD_Y / 2), 130, (255, 255, 255), 2)
    cv2.circle(field, cm2px(FIELD_X / 2, FIELD_Y / 2), 10, (255, 255, 255), -1)

    cv2.ellipse(field, cm2px(0, 0), (50, 50), 0, 0, 90, (255, 255, 255), 2)
    cv2.ellipse(field, cm2px(0, FIELD_Y), (50, 50), 0, 270, 360, (255, 255, 255), 2)
    cv2.ellipse(field, cm2px(FIELD_X, 0), (50, 50), 0, 90, 180, (255, 255, 255), 2)
    cv2.ellipse(field, cm2px(FIELD_X, FIELD_Y), (50, 50), 0, 180, 270, (255, 255, 255), 2)

    cv2.rectangle(field, cm2px(0, 150), cm2px(180, FIELD_Y - 150), (255, 255, 255), 2)
    cv2.rectangle(field, cm2px(FIELD_X - 180, 150), cm2px(FIELD_X, FIELD_Y - 150), (255, 255, 255), 2)

    cv2.rectangle(field, cm2px(0, 250), cm2px(50, FIELD_Y - 250), (255, 255, 255), 2)
    cv2.rectangle(field, cm2px(FIELD_X - 50, 250), cm2px(FIELD_X, FIELD_Y - 250), (255, 255, 255), 2)

    cv2.rectangle(field, cm2px(0, 300), cm2px(-25, FIELD_Y - 300), (255, 255, 255), 2)
    cv2.rectangle(field, cm2px(FIELD_X + 25, 300), cm2px(FIELD_X, FIELD_Y - 300), (255, 255, 255), 2)
    
    return field

def computeDistanceFromBaseline(points_1, points_2, cam_pos_1, cam_pos_2, camera_height, camera_matrix):
    points_1 = np.array(points_1)
    points_2 = np.array(points_2)

    focal_length_x = camera_matrix[0][0]

    disparity = abs(points_1[0] - points_2[0])
    distance_between_cameras = np.sqrt((cam_pos_1[0] - cam_pos_2[0])**2 + (cam_pos_1[1] - cam_pos_2[1])**2)

    depth = (distance_between_cameras * focal_length_x) / disparity
    
    distance_from_baseline = np.sqrt(depth**2 - camera_height**2)
    return distance_from_baseline

if __name__ == "__main__":
    mtx, dist, rvecs, tvecs = calibrateCamera("image/checkerboard/*.jpeg", (8, 6))
    image_0_150 = cv2.imread("image/lapangan/lapangan_0_150.jpg")
    image_0_650 = cv2.imread("image/lapangan/lapangan_0_650.jpg")
    image_600_0 = cv2.imread("image/lapangan/lapangan_600_0.jpg")
    image_600_800 = cv2.imread("image/lapangan/lapangan_600_800.jpg")

    # print("Camera Matrix:\n", mtx)
    epipole_0_150, epipole_0_650 = findEpiPolar(image_0_150, image_0_650)

    rectified_image_0_150, rectified_image_0_650 = rectifyImages(image_0_150, image_0_650, mtx, dist, epipole_0_150, epipole_0_650)

    undistorted_image_600_0 = undistortImage(image_600_0, mtx, dist)
    undistorted_image_600_800 = undistortImage(image_600_800, mtx, dist)
    
    ball_point_0_150 = detectBall(rectified_image_0_150, IMAGE_0_150_YUV)
    ball_point_0_650 = detectBall(rectified_image_0_650, IMAGE_0_650_YUV)
    ball_point_600_0 = detectBall(undistorted_image_600_0, IMAGE_600_0_YUV)
    ball_point_600_800 = detectBall(undistorted_image_600_800, IMAGE_600_800_YUV)

    distance_from_baseline = computeDistanceFromBaseline(ball_point_0_150, ball_point_0_650, CAM_FIELD_0_150, CAM_FIELD_0_650, 140, mtx)
    # print(f"Distance from baseline: {distance_from_baseline} cm")
    
    distance_from_600_0 = computeDistanceFromMiddle(undistorted_image_600_0, CAM_FIELD_600_0, ball_point_600_0, CAM_600_0_MID_FIELD)
    distance_from_600_800 = computeDistanceFromMiddle(undistorted_image_600_800, CAM_FIELD_600_800, ball_point_600_800, CAM_600_800_MID_FIELD, direction=0)
    print(f"Distance from 600 0: {distance_from_600_0} cm")
    print(f"Distance from 600 800: {distance_from_600_800} cm")

    field = drawField()
    field = drawLineFromBaseline(field, CAM_FIELD_0_150, CAM_FIELD_0_650, distance_from_baseline)
    field = drawLineFromBaseline(field, CAM_FIELD_600_0, CAM_FIELD_600_0, distance_from_600_0, direction=0, vertical=False, from_middle=True)
    field = drawLineFromBaseline(field, CAM_FIELD_600_800, CAM_FIELD_600_800, distance_from_600_800, direction=1, vertical=False, from_middle=True)
    
    cv2.namedWindow('Field', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Field', 800, 600)

    # def on_trackbar():
        # pass
    
    # cv2.namedWindow('Trackbars_YUV')
    # cv2.createTrackbar('Y Min', 'Trackbars_YUV', 0, 255, on_trackbar)
    # cv2.createTrackbar('Y Max', 'Trackbars_YUV', 255, 255, on_trackbar)
    # cv2.createTrackbar('U Min', 'Trackbars_YUV', 0, 255, on_trackbar)
    # cv2.createTrackbar('U Max', 'Trackbars_YUV', 255, 255, on_trackbar)
    # cv2.createTrackbar('V Min', 'Trackbars_YUV', 0, 255, on_trackbar)
    # cv2.createTrackbar('V Max', 'Trackbars_YUV', 255, 255, on_trackbar)
    
    # cv2.namedWindow('Image 0 150', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Image 0 150', 800, 600)
    # cv2.namedWindow('Image 0 650', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Image 0 650', 800, 600)
    # cv2.namedWindow('image 600 0', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image 600 0', 800, 600)
    # cv2.namedWindow('image 600 800', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image 600 800', 800, 600)

    # image_0_150_yuv = cv2.cvtColor(rectified_image_0_150, cv2.COLOR_BGR2YUV)
    # image_0_650_yuv = cv2.cvtColor(rectified_image_0_650, cv2.COLOR_BGR2YUV)
    # image_600_0_yuv = cv2.cvtColor(image_600_0, cv2.COLOR_BGR2YUV)
    # image_600_800_yuv = cv2.cvtColor(image_600_800, cv2.COLOR_BGR2YUV)

    # drawPointsOnImage(rectified_image_0_150, ball_point_0_150)
    # drawPointsOnImage(rectified_image_0_650, ball_point_0_650)


    # def mouse_click_callback(event, x, y, flags, param):
        # if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"Click coordinates: (x={x}, y={y})")

    # cv2.setMouseCallback('image 600 0', mouse_click_callback)
    # cv2.setMouseCallback('image 600 800', mouse_click_callback)

    while True:
        # y_min = cv2.getTrackbarPos('Y Min', 'Trackbars_YUV')
        # y_max = cv2.getTrackbarPos('Y Max', 'Trackbars_YUV')
        # u_min = cv2.getTrackbarPos('U Min', 'Trackbars_YUV')
        # u_max = cv2.getTrackbarPos('U Max', 'Trackbars_YUV')
        # v_min = cv2.getTrackbarPos('V Min', 'Trackbars_YUV')
        # v_max = cv2.getTrackbarPos('V Max', 'Trackbars_YUV')

        # print(f"Y Min: {y_min}, Y Max: {y_max}, U Min: {u_min}, U Max: {u_max}, V Min: {v_min}, V Max: {v_max}")

        # image_0_150_final = cv2.inRange(image_0_150_yuv, (y_min, u_min, v_min), (y_max, u_max, v_max))
        # image_0_650_final = cv2.inRange(image_0_650_yuv, (y_min, u_min, v_min), (y_max, u_max, v_max))
        # image_600_0_final = cv2.inRange(image_600_0_yuv, (y_min, u_min, v_min), (y_max, u_max, v_max))
        # image_600_800_final = cv2.inRange(image_600_800_yuv, (y_min, u_min, v_min), (y_max, u_max, v_max))

        # cv2.imshow('Image 0 150', rectified_image_0_150)
        # cv2.imshow('Image 0 650', rectified_image_0_650)
        # cv2.imshow('image 600 0', undistorted_image_600_0)
        # cv2.imshow('image 600 800', undistorted_image_600_800)
        cv2.imshow('Field', field)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()

    
    