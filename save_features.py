import cv2
import dlib
import numpy as np
import csv
import os


path_screenshots = "data/screenshots/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

user = input("user name: ")


while cap.isOpened():
    flag, frame = cap.read()
    k = cv2.waitKey(100)
    if k == ord('p'):
        filename = "./data/user_pic/{}.jpg".format(user)
        cv2.imwrite(filename, frame)
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    frame = cv2.putText(frame, 'Press "p" to get picture ', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)

    # 当前设置仅识别一张人脸，为了便于活体检测
    if len(faces) == 1:
        landmarks = np.mat([[p.x, p.y] for p in predictor(frame, faces[0]).parts()])
        for k, d in enumerate(faces):
            # 计算矩形框大小: Compute the size of rectangle box
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            hh = int(height / 16)
            ww = int(width / 16)
            color_rectangle = (0, 255, 0)
            save_flag = 1
            # 框出人脸范围
            cv2.rectangle(frame, tuple([d.left() - ww, d.top() - hh]), tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)
            frame = cv2.putText(frame, 'Face Detected! ', tuple([d.left() - ww,  d.bottom() + 40]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 标点
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            # cv2.circle(frame, pos, 2, color=(100, 0, 0))
            # cv2.putText(frame, str(idx + 1), None, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)


    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", frame)

cap.release()
cv2.destroyAllWindows()