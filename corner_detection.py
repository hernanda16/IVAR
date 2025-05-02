import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp
from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

def nothing(_):
    pass

def harris_corner_detection(image_path, threshold=0.01):
    # Load the image
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert to float32
    gray = np.float32(gray)

    # Harris corner detection
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # Dilate the result to mark the corners
    dst = cv.dilate(dst, None)

    # Thresholding
    img[dst > threshold * dst.max()] = [0, 0, 255]

    return img
cv.namedWindow('image')
cv.namedWindow('thresh')
cv.createTrackbar('Hue Max', 'image', 0, 255, nothing)
cv.createTrackbar('Hue Min', 'image', 0, 255, nothing)
cv.createTrackbar('Lightness Max', 'image', 0, 255, nothing)
cv.createTrackbar('Lightness Min', 'image', 0, 255, nothing)
cv.createTrackbar('Saturation Max', 'image', 0, 255, nothing)
cv.createTrackbar('Saturation Min', 'image', 0, 255, nothing)
cv.setTrackbarPos('Hue Max', 'image', 255)
cv.setTrackbarPos('Hue Min', 'image', 0)
cv.setTrackbarPos('Lightness Max', 'image', 255)
cv.setTrackbarPos('Lightness Min', 'image', 136)
cv.setTrackbarPos('Saturation Max', 'image', 255)
cv.setTrackbarPos('Saturation Min', 'image', 0)

cv.createTrackbar('Y Max', 'thresh', 0, 255, nothing)
cv.createTrackbar('Y Min', 'thresh', 0, 255, nothing)
cv.createTrackbar('U Max', 'thresh', 0, 255, nothing)
cv.createTrackbar('U Min', 'thresh', 0, 255, nothing)
cv.createTrackbar('V Max', 'thresh', 0, 255, nothing)
cv.createTrackbar('V Min', 'thresh', 0, 255, nothing)
cv.setTrackbarPos('Y Max', 'thresh', 255)
cv.setTrackbarPos('Y Min', 'thresh', 0)
cv.setTrackbarPos('U Max', 'thresh', 255)
cv.setTrackbarPos('U Min', 'thresh', 0)
cv.setTrackbarPos('V Max', 'thresh', 255)
cv.setTrackbarPos('V Min', 'thresh', 0)

def main():
    # Path to the image
    image = cv.imread('test_image.jpg')
    image = cv.resize(image, (400, 300))
    cap_kiri = cv.VideoCapture('Kiri3.mp4')
    cap_kanan = cv.VideoCapture('Kanan3.mp4')

    if not cap_kanan.isOpened():
        print("Error: Could not open video.")
        return
    if not cap_kiri.isOpened():
        print("Error: Could not open video.")
        return

    hls_image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    yuv_image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow('image', hls_image)

    template = np.zeros((1300, 900, 3), np.uint8)
    cv.rectangle(template, (50, 50), (850, 1250), (255, 255, 255), 3)
    cv.ellipse(template, (50, 50), (50, 50), 180, 270, 180, (255, 255, 255), 3, 0)
    cv.ellipse(template, (850, 50), (50, 50), 180, 270, 360, (255, 255, 255), 3, 0)
    cv.rectangle(template, (200, 50), (700, 230), (255, 255, 255), 3)
    cv.rectangle(template, (300, 50), (600, 100), (255, 255, 255), 3)
    cv.line(template, (50, 650), (850, 650), (255, 255, 255), 3)
    cv.circle(template, (450, 650), 130, (255, 255, 255), 3)
    cv.rectangle(template, (200, 1250), (700, 1250 - 180), (255, 255, 255), 3)
    cv.rectangle(template, (300, 1250), (600, 1200), (255, 255, 255), 3)
    cv.ellipse(template, (50, 1250), (50, 50), 180, 180, 90, (255, 255, 255), 3, 0)
    cv.ellipse(template, (850, 1250), (50, 50), 180, 90, 0, (255, 255, 255), 3, 0)
    template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)
    # cv.imshow('Template', template)

    # using namespace cv;
    # Mat frame_lapangan_raw = Mat::zeros(1300, 900, CV_8UC3);

    # rectangle(frame_lapangan_raw, Rect(Point(50, 50), Point(850, 1250)), Scalar(255, 255, 255), 3);

    # ellipse(frame_lapangan_raw, Point(50, 50), Size(50, 50), 180, 270, 180, Scalar(255, 255, 255), 3, 0);
    # ellipse(frame_lapangan_raw, Point(850, 50), Size(50, 50), 180, 270, 360, Scalar(255, 255, 255), 3, 0);
    # rectangle(frame_lapangan_raw, Rect(Point(200, 50), Point(700, 230)), Scalar(255, 255, 255), 3);
    # rectangle(frame_lapangan_raw, Rect(Point(300, 50), Point(600, 100)), Scalar(255, 255, 255), 3);

    # line(frame_lapangan_raw, Point(50, 650), Point(850, 650), Scalar(255, 255, 255), 3);
    # circle(frame_lapangan_raw, Point(450, 650), 130, Scalar(255, 255, 255), 3);

    # rectangle(frame_lapangan_raw, Rect(Point(200, 1250), Point(700, 1250 - 180)), Scalar(255, 255, 255), 3);
    # rectangle(frame_lapangan_raw, Rect(Point(300, 1250), Point(600, 1200)), Scalar(255, 255, 255), 3);
    # ellipse(frame_lapangan_raw, Point(50, 1250), Size(50, 50), 180, 180, 90, Scalar(255, 255, 255), 3, 0);
    # ellipse(frame_lapangan_raw, Point(850, 1250), Size(50, 50), 180, 90, 0, Scalar(255, 255, 255), 3, 0);


    while True:
        ret_kanan, frame_kanan = cap_kanan.read()
        ret_kiri, frame_kiri = cap_kiri.read()

        if not ret_kanan or not ret_kiri:
            cap_kanan.set(cv.CAP_PROP_POS_FRAMES, 0)  # Restart the right video
            cap_kiri.set(cv.CAP_PROP_POS_FRAMES, 0)  # Restart the left video
            ret_kanan, frame_kanan = cap_kanan.read()
            ret_kiri, frame_kiri = cap_kiri.read()

        # cv.imshow('frame_kiri', frame_kiri)
        # cv.imshow('frame_kanan', frame_kanan)

        hue_max = cv.getTrackbarPos('Hue Max', 'image')
        hue_min = cv.getTrackbarPos('Hue Min', 'image')
        lightness_max = cv.getTrackbarPos('Lightness Max', 'image')
        lightness_min = cv.getTrackbarPos('Lightness Min', 'image')
        saturation_max = cv.getTrackbarPos('Saturation Max', 'image')
        saturation_min = cv.getTrackbarPos('Saturation Min', 'image')

        y_max = cv.getTrackbarPos('Y Max', 'thresh')
        y_min = cv.getTrackbarPos('Y Min', 'thresh')
        u_max = cv.getTrackbarPos('U Max', 'thresh')
        u_min = cv.getTrackbarPos('U Min', 'thresh')
        v_max = cv.getTrackbarPos('V Max', 'thresh')
        v_min = cv.getTrackbarPos('V Min', 'thresh')

        # #source coordinates
        # src = np.array([162, 356, 
        #                 776, 415,
        #                 831, 345,
        #                 492, 331,]).reshape((4, 2))
        # #destination coordinates
        # dst = np.array([649, 50, 
        #                 650, 850,
        #                 1249, 850,
        #                 1201, 302,]).reshape((4, 2))

        #source coordinates
        src_kiri = np.array([1075, 592, 
                        999, 505,
                        580, 485,
                        202, 520,]).reshape((4, 2))

        #source coordinates
        src_kanan = np.array([1241, 501, 
                        757, 468,
                        226, 514,
                        48, 655,]).reshape((4, 2))
        #destination coordinates
        dst = np.array([650, 850, 
                        1070, 700,
                        1070, 200,
                        650, 50,]).reshape((4, 2))
        
        # Draw source points on the frame
        for point in src_kanan:
            cv.circle(frame_kanan, tuple(point), 5, (0, 0, 255), -1)
        cv.putText(frame_kanan, 'Source Coordinates', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for point in src_kiri:
            cv.circle(frame_kiri, tuple(point), 5, (0, 0, 255), -1)
        cv.putText(frame_kiri, 'Source Coordinates', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw destination points on the template
        for point in dst:
            cv.circle(template, tuple(point), 5, (0, 255, 0), -1)
        cv.putText(template, 'Destination Coordinates', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Scale destination points to match the template size
        dst_scaled = dst  # Adjust scaling factor as needed

        # Perform projective transformation
        tform_kanan = cv.getPerspectiveTransform(src_kanan.astype(np.float32), dst_scaled.astype(np.float32))
        tf_img_kanan = cv.warpPerspective(frame_kanan, tform_kanan, (template.shape[1], template.shape[0]))

        tform_kiri = cv.getPerspectiveTransform(src_kiri.astype(np.float32), dst_scaled.astype(np.float32))
        tf_img_kiri = cv.warpPerspective(frame_kiri, tform_kiri, (template.shape[1], template.shape[0]))

        # Display the transformed image
        cv.addWeighted(template, 0.5, tf_img_kanan, 0.5, 0, tf_img_kanan)
        # cv.imshow('Transformed Image Kanan', tf_img_kanan)

        cv.addWeighted(template, 0.5, tf_img_kiri, 0.5, 0, tf_img_kiri)
        # cv.imshow('Transformed Image Kiri', tf_img_kiri)

        output = np.zeros_like(template)
        cv.addWeighted(tf_img_kiri, 0.5, tf_img_kanan, 0.5, 0, output)
        cv.imshow('Transformed Image Kiri + Kanan', output)

        # field = cv.inRange(yuv_image, (y_min, u_min, v_min), (y_max, u_max, v_max))
        # cv.imshow('thresh', field)        

        # thresh = cv.inRange(hls_image, (hue_min, lightness_min, saturation_min), (hue_max, lightness_max, saturation_max))
        # cv.imshow('Thresholded Image', thresh)

        # # use canny edge detection
        # edges = cv.Canny(thresh, 100, 200)
        # cv.imshow('Canny Edges', edges)
       
        # Wait for a key press
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == "__main__":
    main()