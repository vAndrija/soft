import cv2
import threading
import numpy as np
import mediapipe as mp
from src.model import create_model

def load_coordinates_from_frame(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        frame.flags.writeable = False

        results = hands.process(frame)

        frame.flags.writeable = True

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        # Rendering results
        coordinates = []
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, )
                for landmark in hand.landmark:
                    coordinates.append(np.asarray([landmark.x * image_width, landmark.y * image_height]))
        ok = False if len(coordinates) > 21 else True
        return coordinates, ok


def get_hand_data_from_frames(frames):
    video = []
    previous_frame = None
    empty_previous_frame = 0
    for frame in frames:

        coordinates, ok = load_coordinates_from_frame(frame)
        if not ok:
            return None
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
    if len(video) == 30:
        return np.asarray(video)
    return None


def webcam():
    cap = cv2.VideoCapture(0)
    clean_frames = 0
    while True:
        frames = []
        threads = []
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        coordinates, ok = load_coordinates_from_frame(framergb)
        if ok and len(coordinates) != 0 and clean_frames==0:
            frames_to_read = 30
            clean_frames = 20
            for i in range(frames_to_read):
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(framergb)
                cv2.imshow("Output", framergb)
                if cv2.waitKey(1) == ord('q'):
                    break
            thread = threading.Thread(target= webcam_test,args=(frames,))
            thread.start()
            threads.append(thread)
        clean_frames-=1
        if clean_frames<0:
            clean_frames = 0
        cv2.imshow("Output", framergb)
        if cv2.waitKey(200) == ord('q'):
            break
    for thread in threads:
        thread.join()
    cap.release()
    cv2.destroyAllWindows()


def webcam_test(frames):
    model = create_model()
    model.load_weights("../model/weights3.h5")
    hand_data = get_hand_data_from_frames(frames)
    if hand_data is not None:
        result = model.predict(np.asarray([hand_data]), batch_size=1, verbose=0)
        result = result[0].tolist()
        max_value = max(result)
        index = result.index(max_value)
        print(index)

if __name__ == '__main__':
    webcam()
