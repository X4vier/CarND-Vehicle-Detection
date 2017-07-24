import numpy as np
import cv2
from search_image2 import find_cars
from search_image2 import draw_boxes
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cap = cv2.VideoCapture("project_video.mp4")

window_history = {}
frame_history = {}
num_frames = 3
threshold = 10
frame_number = 0
start_frame = 200
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame_number<start_frame:
        frame_number += 1
        continue
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        frame_history[frame_number] = frame
        windows = find_cars(img)
        window_history[frame_number] = find_cars(img)
        test = draw_boxes(img, windows, random_color=True)
        plt.imshow(test)
        plt.show()
        if frame_number >= start_frame + num_frames-1:
            windows = []
            [windows.extend(window_history[x]) for x in range (frame_number-num_frames+1, frame_number+1)]
            middle_frame = frame_history[frame_number-num_frames//2]
            heatmap = np.zeros_like(middle_frame[:, :, 0]).astype(np.float)

            for window in windows:
                # Add += 1 for all pixels inside each bbox
                heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1


            heatmap[heatmap <= threshold] = 0

            labels = label(heatmap)
            bboxes = []
            for car_number in range(1, labels[1] + 1):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Define a bounding box based on min/max x and y
                bboxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

            out_img = draw_boxes(middle_frame, window_history[frame_number-num_frames//2])
            cv2.imwrite("output_images/{}.jpg".format(frame_number-num_frames//2), out_img)
            frame_history.pop(frame_number-num_frames+1)
            window_history.pop(frame_number-num_frames+1)
            print(len(frame_history))
        frame_number += 1

    else:
        cap.release()


# out = cv2.VideoWriter('out_video.avi', -1, 24, (1280, 720))

# for img in new_frames:
#     out.write(img)


cv2.destroyAllWindows()
# out.release()
