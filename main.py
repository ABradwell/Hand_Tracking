"""
    Author: Aiden Stevenson Bradwell
    Date: 2021-11-22
    Affiliation: University of Ottawa, Ottawa, Ontario (Student)

    Description:

    Libraries required:
        opencv-python
        mediapipe
        tensorflow

    Referenced sites:
    https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
"""
import math

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


def get_closest_hand(point_list, new_x, new_y):
    last_hand_1 = point_list[0][-1]
    last_hand_2 = point_list[1][-1]

    x_diff_1 = abs(last_hand_1[0] - new_x)
    y_diff_1 = abs(last_hand_1[1] - new_y)
    x_diff_2 = abs(last_hand_2[0] - new_x)
    y_diff_2 = abs(last_hand_2[1] - new_y)

    distance_1 = math.sqrt(pow(x_diff_1, 2) + pow(y_diff_1, 2))
    distance_2 = math.sqrt(pow(x_diff_2, 2) + pow(y_diff_2, 2))

    if distance_1 < distance_2:
        return 0
    else:
        return 1


def track_method(frame, hands, pts):
    """

    :param frame:
    :param hands:
    :param pts:
    :return:
    """

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    h, w, c = frame.shape  # Take image shape (used for normalized points)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    found = hands.process(frame_rgb)  # Call pre-trained model with new frame

    # There can be two hands tracked.
    #   If only one is detected, it will be drawn blue.
    #   If two are found one will be purple.
    # new_points stores the potential new points. If found, they will be added to the point list of the appropriate hand
    new_points = [[-1, -1], [-1, -1]]
    hand_colors = [(255, 0, 0), (255, 0, 255)]

    # If a hand has been found
    if found.multi_hand_landmarks:

        # Inform user of the 3 or more hands
        if len(found.multi_hand_landmarks) > 2:
            print("More than two hands detected. Only tracking first two")

        # For both hands (or one hand if applicable)
        for hand_idx in range(0, min(2, len(found.multi_hand_landmarks))):
            handslms = found.multi_hand_landmarks[hand_idx]

            # Take the values of all landmark coordinates into two lists
            x_vals = []
            y_vals = []
            for lm in handslms.landmark:
                #De-normalize corodinates
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)

                x_vals.append(lmx)
                y_vals.append(lmy)

            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)

            # Draw bounding box around the hand being read
            cv2.rectangle(frame, (min(x_vals), min(y_vals)), (max(x_vals), max(y_vals)), (0, 255, 0), 4)

            # Calculate the palm of the hand
            palm_center_x = int((x_vals[9] + x_vals[13]) / 2)
            palm_center_y = int((y_vals[9] + y_vals[13]) / 2)

            # Draw each point in the movement-tracer
            for pt in range(1, len(pts[hand_idx])):
                if continue_line(pts[hand_idx][pt-1], pts[hand_idx][pt]):
                    cv2.line(frame, pts[hand_idx][pt-1], pts[hand_idx][pt], hand_colors[hand_idx], 2)


            # hand_point_index = get_closest_hand(pts, palm_center_x, palm_center_y)
            # Returned later, added to tracer-list in the VideoShower class
            # new_points[hand_point_index] = [palm_center_x, palm_center_y]
            new_points[hand_idx] = [palm_center_x, palm_center_y]

    return frame, new_points[0], new_points[1]


def continue_line(last_point, new_point):
    if last_point == [None, None]:
        return True

    return abs(last_point[0] - new_point[0]) < 75 and abs(last_point[1] - new_point[1]) < 75


if __name__ == "__main__":

    # Initialize webcam
    capture = cv2.VideoCapture("videos/videos_partB.mp4")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_length = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    img_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    incap = cv2.VideoWriter("videos/result.avi", fourcc, fps, (img_width, img_height))



    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    # This model is a pretrained model by **,
    # found @ https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
    model = load_model('mp_hand_gesture')
    points = [[], []]

    while capture.isOpened():

        ret, frame = capture.read()
        if not ret:
            break

        image, new_h1, new_h2 = track_method(frame, hands, points)

        # If one hand was detected, add new point to movement-tracer
        if new_h1 != [-1, -1]:
            points[0].append(new_h1)
            if len(points[0]) > 75:
                points[0].pop(0)

        # If a second hand was detected, add new point to movement-tracer
        if new_h2 != [-1, -1]:
            points[1].append(new_h2)
            if len(points[1]) > 75:
                points[1].pop(0)

        # Display marked image
        cv2.imshow("Traced", image)
        cv2.waitKey(1)

        # Write to output
        incap.write(np.array(image))

    capture.release()
    incap.release()
    cv2.destroyAllWindows()
