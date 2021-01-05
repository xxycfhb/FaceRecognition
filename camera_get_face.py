import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


path_screenshots = "data/screenshots/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)


# 眨眼
def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 水平
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# 张嘴
def mouth_aspect_ratio(mouth):
    # 垂直点位
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


# 点头
def nod_aspect_ratio(size, pre_point, now_point):
    return abs(float((pre_point[1] - now_point[1]) / (size[0] / 2)))



# 摇头
def shake_aspect_ratio(size, pre_point, now_point):
    return abs(float((pre_point[0] - now_point[0]) / (size[1] / 2)))




# 活体检测常量
EYE_AR_THRESH = 0.2         # 眨眼阈值
EYE_AR_CONSEC_FRAMES = 2    # 眨眼帧数
MAR_THRESH = 0.5            # 张嘴阈值
MOUTH_AR_CONSEC_FRAMES = 3  # 张嘴帧数
NOD_THRESH = 0.03           # 点头阈值
NOD_CONSEC_FRAMES = 5       # 点头帧数
SHAKE_THRESH = 0.03         # 摇头阈值
SHAKE_CONSEC_FRAMES = 5     # 摇头帧数
COUNT = 0
TOTAL = 0                   # 成功眨眼次数
mCOUNT = 0
mTOTAL = 0                  # 成功张嘴次数
nCOUNT = 0
nTOTAL = 0                  # 成功点头次数
sCOUNT = 0
sTOTAL = 0                  # 成功摇头次数
r_eye_ear = 0
l_eye_ear = 0
compare_point = [0, 0]

while cap.isOpened():
    flag, frame = cap.read()
    k = cv2.waitKey(100)
    if k == ord('q'):
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    frame = cv2.putText(frame, 'Press "q" to quit! ', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1,
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

        # 活体检测
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:41]
        mouth_points = landmarks[48:68]
        nose_points = landmarks[32:36]
        nose_center = nose_points.mean(axis=0).tolist()[0]
        noseHull = cv2.convexHull(nose_points)

        size = frame.shape

        l_eye_ear = eye_aspect_ratio(left_eye)
        r_eye_ear = eye_aspect_ratio(left_eye)
        t_ear = (l_eye_ear + r_eye_ear) / 2.0
        mouth_ear = mouth_aspect_ratio(mouth_points)
        nod_value, shake_value = 0, 0
        if compare_point[0] != 0 and compare_point[0] != 0:
            nod_value = nod_aspect_ratio(size, nose_center, compare_point)
            shake_value = shake_aspect_ratio(size, nose_center, compare_point)
        compare_point = nose_center

        if t_ear < EYE_AR_THRESH:
            COUNT += 1
        else:
            if COUNT >= EYE_AR_CONSEC_FRAMES:
                # 眨眼
                TOTAL += 1
            COUNT = 0
        if mouth_ear > MAR_THRESH:
            mCOUNT += 1
        else:
            if mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
                # 张嘴
                mTOTAL += 1
            mCOUNT = 0
        if nod_value > NOD_THRESH:
            nCOUNT += 1
        else:
            if nCOUNT >= NOD_CONSEC_FRAMES:
                # 点头
                nTOTAL += 1
            nCOUNT = 0
        if shake_value > SHAKE_THRESH:
            sCOUNT += 1
        else:
            if sCOUNT >= SHAKE_CONSEC_FRAMES:
                # 摇头
                sTOTAL += 1
            sCOUNT = 0

        sum = 0

        # 眨眼成功，报出信息
        if TOTAL > 0:
            sum += 1
            frame = cv2.putText(frame, 'Eyes moved', tuple([d.left() - ww, d.bottom() + 60]),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        # 张嘴成功，报出信息
        if mTOTAL > 0:
            sum += 1
            frame = cv2.putText(frame, 'Mouse moved', tuple([d.left() - ww, d.bottom() + 80]),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        # 点头成功，报出信息
        if nTOTAL > 0:
            sum += 1
            frame = cv2.putText(frame, 'Head nodded', tuple([d.left() - ww, d.bottom() + 100]),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        # 摇头成功，报出信息
        if sTOTAL > 0:
            sum += 1
            frame = cv2.putText(frame, 'Head shaked', tuple([d.left() - ww, d.bottom() + 120]),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        # 当四种检测机制有三种及以上成功时，通过验活机制
        if sum > 2:
            frame = cv2.putText(frame, 'Live detection succeed!', tuple([d.left() - ww, d.bottom() + 150]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        frame = cv2.putText(frame, 'Sorry, please try again! ', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
        TOTAL = 0
        mTOTAL = 0
        nTOTAL = 0
        sTOTAL = 0

    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", frame)

cap.release()
cv2.destroyAllWindows()