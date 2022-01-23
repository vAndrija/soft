import numpy as np
import cv2  # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
from tensorflow import keras
import mediapipe as mp
import os
import csv


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
        plt.show()


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def load_coordinates_from_frame(path):
    image = load_image(path)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        # Rendering results
        coordinates = []
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, )

                for landmark in hand.landmark:
                    coordinates.append(np.asarray([landmark.x * image_width, landmark.y * image_height]))
                    # coordinates.append(landmark.y * image_height)
                # print('hand_landmarks:', hand)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
                #     f'{hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})'
                # )

        # display_image(image)
        ok = False if len(coordinates) > 21 else True
        return coordinates, ok


def load_train_data(csv_file_path, base_path):
    train_data = []
    train_labels = []
    c = 0
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=";")
        for row in csv_reader:
            c += 1
            previous_frame = None
            empty_previous_frame = 0
            video = []
            extend_video = True
            for frame in sorted(os.listdir(base_path + row[0])):
                coordinates, ok = load_coordinates_from_frame(base_path + row[0] + "/" + frame)
                if not ok:
                    extend_video = False
                    break
                if len(coordinates) == 0:
                    if previous_frame is None:
                        empty_previous_frame += 1
                    else:
                        coordinates = previous_frame
                        video.append(coordinates)
                else:
                    if empty_previous_frame == 0:
                        previous_frame = coordinates
                        video.append(coordinates)
                    else:
                        previous_frame = coordinates
                        for i in range(empty_previous_frame + 1):
                            video.append(coordinates)
                        empty_previous_frame = 0
            if extend_video and len(video) == 30:
                print(c)
                train_data.append(np.asarray(video))
                train_labels.append(int(row[2]))
    return np.asarray(train_data), np.asarray(train_labels)


if __name__ == '__main__':
    load_train_data("../data/train.csv", "../data/train/")
