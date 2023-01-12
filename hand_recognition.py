import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math
import tkinter as tk

import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 先读入图像
img = cv2.imread("D://E.jpg")
# 将3倍这个尺寸设置为初始尺寸
height, weight = img.shape[0]*3, img.shape[1]*3

# 用一个全局变量来记录当前播放的视力表字母的方向
cur_direct = ''

# 需要保存图片的一个缩放比例，方便后面进行结果的展示
ratio = 1

# 需要初始化得到屏幕尺寸
root = tk.Tk()
center_x, center_y = root.winfo_screenwidth()//2,root.winfo_screenheight()//2

# 将左右手进行一个区分展示
def get_label(index, hand, results):
    output = None
    # 如果屏幕当中没有手的话将会返回一个空对象
    for idx, classification in enumerate(results.multi_handedness):
        # 找到分类对应的手的index
        if classification.classification[0].index == index:
            
            # 获得分类的结果，然后进行输出的格式化
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # 要获得坐标, 不是百分比
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            # 返回文本以及对应的坐标
            output = text, coords
            
    return output



# 计算方位角函数
def azimuthAngle(x1,  y1,  x2,  y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


# 判断所指的方向，顺便将方位角转化为对应的方向信号
def get_direction(hand):
    # 左右手的处理是一样的
    forward = hand.landmark[8]
    backward = hand.landmark[5]

    coord1 = tuple(np.multiply(
                np.array((forward.x, forward.y)),
            [640,480]).astype(int))

    coord2 = tuple(np.multiply(
                np.array((backward.x, backward.y)),
            [640,480]).astype(int))

    # 为了防止除以0等情况的发生，先对数据进行一下处理
    degree = azimuthAngle(coord1[0], coord1[1], coord2[0], coord2[1])

    direct = -1
    if (degree > 0 and degree < 20) or (degree > 340 and degree < 360):
        direct = 'Up'
    elif degree > 70 and degree < 110:
        direct = 'Left'
    elif degree > 160 and degree < 200:
        direct = 'Down'
    elif degree > 250 and degree < 290:
        direct = 'Right'
        
    return direct

out_win = 'please direct!'

# 改变方向进行展示
def show_E():
    # 随机生成一个比例，然后保存到ratio当中，从0-1的均匀分布中采样
    global ratio 
    ratio = round(np.random.uniform(0.15, 1), 2)
    temp_h, temp_w = int(height*ratio), int(weight*ratio)
    cur_img = cv2.resize(img, (temp_h, temp_w), interpolation=cv2.INTER_CUBIC)

    # 并进行旋转操作,原方向E朝右，该旋转列表对应为 上，左，下，右
    directs = ['Up', 'Left', 'Down', 'Right']
    rotate_list = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, 0]
    cur_index = np.random.randint(0, 4)
    global cur_direct
    cur_direct = directs[cur_index]
    if cur_index != 3:
        cur_img = cv2.rotate(cur_img, rotate_list[cur_index]) # 这个下标代表的就是方向代号
    
    
    # 将窗口显示在图片中间，用到两个工具包
    temp_x, temp_y = center_x - temp_w//2, center_y - temp_h//2

    cv2.imshow(out_win, cur_img)
    cv2.moveWindow(out_win, temp_x, temp_y)


# 要相应的有颜色反馈
def show_result(result:int):

    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    temp_h, temp_w = int(height*ratio), int(weight*ratio)
    # cv2.resizeWindow("result", temp_h, temp_w)
    temp_x, temp_y = center_x - temp_w//2, center_y - temp_h//2

    # result传过来1代表正确，传过来2代表不正确
    # mat = np.zeros([temp_h, temp_w, 3], np.uint8)
    # mat[:,:,result] = np.ones([temp_h, temp_w]) * 255

    # cv2.imshow("result", mat)
    # 摇头
    for i in range(10):
        if i % 2 == 0:
            cv2.moveWindow(out_win, temp_x + 20, temp_y)
        else:
            cv2.moveWindow(out_win, temp_x - 20, temp_y)
        time.sleep(0.05)

    # time.sleep(0.5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    


# 识别手部动作
def main():
    cap = cv2.VideoCapture(0)
    delay = 0

    # 设置几个信号量
    multi_hand = 0
    show_count = 0
    pre_direct = ''

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            ret, frame = cap.read()
            
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
            # Flip on horizontal
            image = cv2.flip(image, 1)           
            # Set flag
            image.flags.writeable = False            
            # 检测当前帧当中是否有手
            results = hands.process(image)            
            # Set flag to true
            image.flags.writeable = True            
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)            
            # print(results)
            
            # 如果不为空
            if results.multi_hand_landmarks:
                # 循环体是每一只手
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )                   
                    # 判断是左手还是右手
                    text = ''
                    if get_label(num, hand, results):
                        text, coord = get_label(num, hand, results)
                        # 在关节点0处输出左右手信息
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # 提示单手操作
                    if len(list(enumerate(results.multi_hand_landmarks))) > 1:
                        cv2.putText(image, "Please use one hand!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # 进行方向判断，需要满足两个判断条件
                    # 但是也不需要每一帧图像都去判断，要有一个间隔                   
                    if len(list(enumerate(results.multi_hand_landmarks))) == 1:
                        show_count += 1
                        direct = get_direction(hand)
                        # 根据方向进行相应输出
                        if direct != -1:
                            cv2.putText(image, direct, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # 要保证是一个动作保持
                        if direct != pre_direct:
                            pre_direct = direct
                            show_count = 0

                        if show_count == 15:
                            # 要判断方向是否一致
                            # print(cur_direct, direct)
                            # print('ratio', ratio)
                            if cur_direct != direct:
                                show_result(1)
                        
                            show_E()
                            show_count = 0


            # 保存图片   
            #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
            cv2.imshow('Hand Tracking', image)
            cv2.moveWindow('Hand Tracking', 10, 500)


            # 判断是否符合退出条件ESC
            if cv2.waitKey(10) & 0xFF == 27:
                break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_E()
    main()
    # show_result(2)
    # for i in range(10):
    #     show_E()


