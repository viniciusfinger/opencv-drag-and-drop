import cv2
from HandDetector import HandDetector

COLOR_PINK = (255, 0, 255)
COLOR_GREEN = (0, 255, 0)
RECTANGLE_SIZE = 200

rectangle_color = COLOR_PINK
rectangle_x, rectangle_y = 100, 100

webcam_capture = cv2.VideoCapture(1)
webcam_capture.set(3, 1280)
webcam_capture.set(4, 720)

hand_detector = HandDetector()

def is_indicator_in_rectangle():
    return rectangle_x < indicator_landmark[1] < rectangle_x + RECTANGLE_SIZE and rectangle_y < indicator_landmark[2] < rectangle_y + RECTANGLE_SIZE

while True:
    success, img = webcam_capture.read()

    img = cv2.flip(img, 1)

    img = hand_detector.process_hands(img)
    landmark_list = hand_detector.get_positions(img)

    if landmark_list:
        indicator_landmark = landmark_list[8]
        print("Cursor 1 (X ou width): ", indicator_landmark[1])
        print("Cursor 2 (Y or height): ", indicator_landmark[2])
        
        if is_indicator_in_rectangle():
            rectangle_color = COLOR_GREEN
            if hand_detector.is_click(landmark_list):
                rectangle_x, rectangle_y = indicator_landmark[1] - 100, indicator_landmark[2] - 100
        else:
            rectangle_color = COLOR_PINK
        
    cv2.rectangle(img, (rectangle_x, rectangle_y), (rectangle_x + RECTANGLE_SIZE, rectangle_y + RECTANGLE_SIZE), rectangle_color, cv2.FILLED)

    cv2.imshow("Drag and Drop OpenCV", img)
    
    cv2.waitKey(1) & 0xFF == ord('q')
