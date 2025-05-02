import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp
from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

model = YOLO('best_fit.pt')

def nothing(_):
    pass

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
    cap_kiri = cv.VideoCapture('Kiri3.mp4')
    cap_kanan = cv.VideoCapture('Kanan3.mp4')

    if not cap_kanan.isOpened():
        print("Error: Could not open video.")
        return
    if not cap_kiri.isOpened():
        print("Error: Could not open video.")
        return
    
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

    while True:
        ret_kanan, frame_kanan = cap_kanan.read()
        ret_kiri, frame_kiri = cap_kiri.read()

        if not ret_kanan or not ret_kiri:
            cap_kanan.set(cv.CAP_PROP_POS_FRAMES, 0)  # Restart the right video
            cap_kiri.set(cv.CAP_PROP_POS_FRAMES, 0)  # Restart the left video
            ret_kanan, frame_kanan = cap_kanan.read()
            ret_kiri, frame_kiri = cap_kiri.read()

        result_kiri = model(frame_kiri)[0]
        result_kanan = model(frame_kanan)[0]

        ball_kiri = [0,0]
        ball_kanan = [0,0]
        
        if(result_kiri.boxes is not None):
            boxes_kiri = result_kiri.boxes.xyxy.cpu().numpy()
            for box in boxes_kiri:
                x1, y1, x2, y2 = map(int, box)
                ball_kiri[0] = (x1 + x2) // 2
                ball_kiri[1] = y2
                # cv.circle(frame_kiri, ((x1 + x2) // 2, y2), 5, (0, 255, 0), -1)
        if(result_kanan.boxes is not None):
            boxes_kanan = result_kanan.boxes.xyxy.cpu().numpy()
            for box in boxes_kanan:
                x1, y1, x2, y2 = map(int, box)
                ball_kanan[0] = (x1 + x2) // 2
                ball_kanan[1] = y2
                # cv.circle(frame_kanan, ((x1 + x2) // 2, y2), 5, (0, 255, 0), -1) 

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

        map_frame = template.copy()
        # Transform ball points using homography if detected
        ball_kiri_homography = None
        ball_kanan_homography = None

        ball_is_out_of_bounds = True

        if ball_kiri != [0, 0]:
            ball_kiri_homography = cv.perspectiveTransform(np.array([[ball_kiri]], dtype=np.float32), tform_kiri)[0][0]
        if ball_kanan != [0, 0]:
            ball_kanan_homography = cv.perspectiveTransform(np.array([[ball_kanan]], dtype=np.float32), tform_kanan)[0][0]

        if ball_kiri_homography is not None and ball_kanan_homography is not None:
            if (ball_kiri_homography[0] > 50 - 15 and ball_kiri_homography[0] < 850 + 15 and ball_kiri_homography[1] > 50 - 15 and ball_kiri_homography[1] < 1250 + 15
            and ball_kanan_homography[0] > 50 - 15 and ball_kanan_homography[0] < 850 + 15 and ball_kanan_homography[1] > 50 - 15 and ball_kanan_homography[1] < 1250 + 15):
                ball_is_out_of_bounds = False
        elif ball_kiri_homography is not None:
            if ball_kiri_homography[0] > 50 - 15 and ball_kiri_homography[0] < 850 + 15 and ball_kiri_homography[1] > 50 - 15 and ball_kiri_homography[1] < 1250 + 15:
                ball_is_out_of_bounds = False
        elif ball_kanan_homography is not None:
            if ball_kanan_homography[0] > 50 - 15 and ball_kanan_homography[0] < 850 + 15 and ball_kanan_homography[1] > 50 - 15 and ball_kanan_homography[1] < 1250 + 15:
                ball_is_out_of_bounds = False

        # Calculate combined ball position if both are detected
        if ball_kiri_homography is not None and ball_kanan_homography is not None:
            combined_ball = (ball_kiri_homography + ball_kanan_homography) / 2
            cv.circle(map_frame, (int(combined_ball[0]), int(combined_ball[1])), 15, (255, 0, 255), -1)
            cv.circle(tf_img_kiri, (int(combined_ball[0]), int(combined_ball[1])), 15, (255, 0, 255), -1)
            cv.circle(tf_img_kanan, (int(combined_ball[0]), int(combined_ball[1])), 15, (255, 0, 255), -1)
        elif ball_kiri_homography is not None:
            cv.circle(map_frame, (int(ball_kiri_homography[0]), int(ball_kiri_homography[1])), 15, (255, 0, 0), -1)
            cv.circle(tf_img_kiri, (int(ball_kiri_homography[0]), int(ball_kiri_homography[1])), 15, (255, 0, 0), -1)
            cv.circle(tf_img_kanan, (int(ball_kiri_homography[0]), int(ball_kiri_homography[1])), 15, (255, 0, 0), -1)
        elif ball_kanan_homography is not None:
            cv.circle(map_frame, (int(ball_kanan_homography[0]), int(ball_kanan_homography[1])), 15, (0, 0, 255), -1)
            cv.circle(tf_img_kiri, (int(ball_kanan_homography[0]), int(ball_kanan_homography[1])), 15, (0, 0, 255), -1)
            cv.circle(tf_img_kanan, (int(ball_kanan_homography[0]), int(ball_kanan_homography[1])), 15, (0, 0, 255), -1)

        if(ball_is_out_of_bounds):
            cv.putText(map_frame, 'Ball is out of bounds', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Inverse homography from output to frame_kanan and frame_kiri
        inv_tform_kanan = cv.getPerspectiveTransform(dst_scaled.astype(np.float32), src_kanan.astype(np.float32))
        inv_tform_kiri = cv.getPerspectiveTransform(dst_scaled.astype(np.float32), src_kiri.astype(np.float32))

        # Warp the output back to the original frames
        inv_frame_kanan = cv.warpPerspective(tf_img_kanan, inv_tform_kanan, (frame_kanan.shape[1], frame_kanan.shape[0]))
        inv_frame_kiri = cv.warpPerspective(tf_img_kiri, inv_tform_kiri, (frame_kiri.shape[1], frame_kiri.shape[0]))

        # Display the inverse transformed frames
        # cv.imshow('frame_kiri', frame_kiri)
        # cv.imshow('frame_kanan', frame_kanan)

        map_frame = cv.resize(map_frame, (int(template.shape[1] * 0.5), int(template.shape[0] * 0.5)))
        inv_frame_kiri = cv.resize(inv_frame_kiri, (int(template.shape[1] * 0.5), int(template.shape[0] * 0.5)))
        inv_frame_kanan = cv.resize(inv_frame_kanan, (int(template.shape[1] * 0.5), int(template.shape[0] * 0.5)))

        combined_frame = np.hstack((inv_frame_kiri, map_frame, inv_frame_kanan))
        cv.imshow('Combined Frame', combined_frame)

        # cv.imshow('Map Frame', map_frame)
        # cv.imshow('Inverse Transformed Frame Kanan', inv_frame_kanan)
        # cv.imshow('Inverse Transformed Frame Kiri', inv_frame_kiri)

        output = np.zeros_like(template)
        cv.addWeighted(tf_img_kiri, 0.5, tf_img_kanan, 0.5, 0, output)
        # cv.imshow('Transformed Image Kiri + Kanan', output)
       
        # Wait for a key press
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == "__main__":
    main()