from robomaster import robot
import cv2  # 引入OpenCV模块
import numpy as np
import pytesseract as tess  # OCR识别模块
import robomaster.config
from PIL import Image
import time
import sys

def pretreatment(img,C=1.1,L=100,S=40,k=1):
    # 亮度和对比度调节
    # C是对比度，应在0~3之间调节（1为不做调整）
    # L是亮度，应在0~255之阿调节（0为不做调整）
    img = np.uint8(np.clip((C * img + L), 0, 255))
    n = S/100
    # 图像锐化
    # s为锐化程度，应在0~100之间调节（0为不做调整），k为锐化卷卷积核形式(在1和2之间选择)
    if k == 1:
        sharpen_op = np.array([[0, -n, 0],
                               [-n, 4 * n + 1, -n],
                               [0, -n, 0]], dtype=np.float32)
    if k == 2:
        sharpen_op = np.array([[-n, -n, -n],
                               [-n, 8 * n + 1, -n],
                               [-n, -n, -n]], dtype=np.float32)
    img = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    img = cv2.convertScaleAbs(img)
    return img

def flight_start():
    # 开启摄像头检测
    cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi', 640, 480)
    cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('picture', 640, 480)
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("low")
    tl_camera.set_resolution("high")
    tl_camera.set_bitrate(6)
    img = tl_camera.read_cv2_image(strategy="newest")
    # 无人机起飞

    tl_flight.takeoff().wait_for_completed()


# *********
# 函数功能：输入一个图片，识别指定的字母
# 输入：text_input：需要识别的字母,img：需要识别的图片
# 输出：①是否识别到的标志位②所识别字母的中心坐标
# *********

def morphological_Processing(img):
    # 进行闭运算,10*10为卷积核大小,MORPH_ELLIPSE表示创建卷积核形状为椭圆
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close_operation = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=2)
    # 进行两轮开运算，使形状更平滑便于识别
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT,(10,1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT,(1,10))
    open_operationX = cv2.morphologyEx(close_operation, cv2.MORPH_OPEN, kernelX, iterations=1)
    open_operationY = cv2.morphologyEx(open_operationX, cv2.MORPH_OPEN, kernelY, iterations=1)
    return open_operationY

def Congnization_char(text_input, img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV格式
    # ************
    # 红色阈值处理
    # ************
    lower_hsv1 = np.array([0, 43, 46])
    upper_hsv1 = np.array([10, 255, 255])
    lower_hsv2 = np.array([156, 43, 46])
    upper_hsv2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_hsv1, upper_hsv1)  # 根据红色阈值二值化
    mask2 = cv2.inRange(img_hsv, lower_hsv2, upper_hsv2)  # 根据红色阈值二值化
    mask_red = mask1 + mask2  # 将红色的两部分阈值进行相加
    mask_red = cv2.bitwise_not(mask_red)  # 将图像像素反转（这里其实可以设定ROI感兴趣区为目标图像区域，然后反转）
    cv2.imshow("mask", mask_red)
    cv2.waitKey(1)
    # *************
    # 边缘检测获得轮廓
    # *************
    canny = cv2.Canny(mask_red, 80, 160, 3)  # canny边缘处理
    cv2.imshow("canny", canny)
    cv2.waitKey(1)
    countours, hierarchy = cv2.findContours(canny, mode=0, method=2)  # 获取边缘的坐标和数目
    # ************
    # 获取最大轮廓
    # ************
    area = []
    for i in range(len(countours)):
        area.append(cv2.contourArea(countours[i]))  # 获取各个边缘的面积
    if area:
        print("获取到边缘")
        max_idx = np.where(area == np.max(area))  # 找到最大面积边缘的索引
        max_rect = cv2.boundingRect(countours[max_idx[0][0]])  # 得到最大面积边缘的矩形框
    else:
        print("未获取到边缘")
        max_rect = [1, 1, 10, 10]  # 得到最大面积边缘的矩形框
    # ************
    # 对最大轮廓进行分割（设定感兴趣区)
    # ************
    [x, y, w, h] = max_rect
    center_graph = [x + w / 2, y + h / 2]
    my_roi = mask_red[y - 10:y + h + 10, x - 10:x + w + 10]
    try:
        cv2.imshow("roi", my_roi)
    except:
        pass
    textImage = Image.fromarray(my_roi)
    try:
        text = tess.image_to_string(textImage, lang="eng", config=r'--psm 10')  # 使用ocr进行识别
    except:
        text = " "
    print("识别结果: %s" % text)
    if text_input in text:
        print("识别结果: %s" % text)
        return True, center_graph
    return False, center_graph


def Say_hello():
    text_ni = '00r0r0000r00rrr0rr0r00r00r000r000r0r0rr00rr00rr00r00rr0000000000'
    test_hao = '0r0rrrr0rrr000r0r0r00r00r0rrrrr00r000r00r0r00r00r0r0rr0000000000'
    tl_drone.led.set_mled_graph(text_ni)
    time.sleep(3)
    tl_drone.led.set_mled_graph(test_hao)
    time.sleep(3)


def Across_arch():
    tl_flight.up(distance=90).wait_for_completed()
    tl_flight.rotate(angle=53).wait_for_completed()
    tl_flight.forward(distance=500).wait_for_completed()


def across_anch():
    # 寻找拱门的标志
    tl_flight.up(distance=80).wait_for_completed()
    stack.append(490)
    tl_flight.rotate(angle=25).wait_for_completed()
    stack.append(325)
    anch_find = False
    anch_x = False
    anch_y = False
    for j in range(0, 4):
        for i in range(0, 20):
            img = tl_camera.read_cv2_image(strategy="newest")
            # img = pretreatment(img,C=1.1,L=90,S=40,k=1)
            center_x, center_y = img.shape[:2]
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV格式
            # 设定颜色阈值,
            lower_hsv1 = np.array([0, 43, 46])
            upper_hsv1 = np.array([10, 255, 255])
            lower_hsv2 = np.array([156, 43, 46])
            upper_hsv2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_hsv1, upper_hsv1)  # 根据红色阈值二值化
            mask2 = cv2.inRange(img_hsv, lower_hsv2, upper_hsv2)  # 根据红色阈值二值化
            mask_red = mask1 + mask2  # 将红色的两部分阈值进行相加
            mask_red = morphological_Processing(mask_red)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
            # mask_red = cv2.dilate(mask_red, kernel)
            mask_red = cv2.bitwise_not(mask_red)  # 将图像像素反转（这里其实可以设定ROI感兴趣区为目标图像区域，然后反转）
            cv2.imshow("mask", mask_red)
            cv2.waitKey(20)
            canny = cv2.Canny(mask_red, 80, 160, 2)  # canny边缘处理
            cv2.imshow("canny", canny)
            cv2.waitKey(20)
            countours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, method=2)  # 获取边缘的坐标和数目
            area = []
            for i in range(len(countours)):
                area.append(cv2.contourArea(countours[i]))  # 获取各个边缘的面积
            try:
                max_idx = np.where(area == np.max(area))  # 找到最大面积边缘的索引
                max_area = np.max(area)
                print(max_area)
            except:
                continue
            max_rect = cv2.boundingRect(countours[max_idx[0][0]])  # 得到最大面积边缘的矩形框
            cv2.rectangle(img, max_rect, (0, 0, 225), 2, 8, 0)  # 在原图像中画出矩形框
            cv2.rectangle(mask_red, max_rect, (0, 0, 225), 2, 8, 0)
            [x, y, w, h] = max_rect
            drone_x = x + w / 2
            drone_y = y + h / 2
            my_roi = mask_red[y:y + h, x:x + w]
            cv2.imshow("origingal", img)
            cv2.waitKey(20)
            try:
                cv2.imshow("ROI", my_roi)
            except:
                pass
            # ********************
            # 寻找拱门
            # ********************
            if max_area > 15000:
                print("找到拱门")
                anch_find = True
                if anch_find:
                    if drone_x < center_x * 7 / 16:  # 拱门在无人机左侧
                        print("左转")
                        tl_flight.rotate(angle=-3).wait_for_completed()
                        stack.append(205)
                    elif drone_x > center_x * 10 / 16:  # 拱门在无人机右侧
                        print("右转")
                        tl_flight.rotate(angle=3).wait_for_completed()
                        stack.append(305)
                    else:
                        print("左右位置合适")
                        anch_x = True
                    # if drone_y < center_y * 3 / 8:  # 拱门在无人机上侧
                    #     print("上升")
                    #     tl_flight.up(distance=20).wait_for_completed()
                    #     stack.append(420)
                    # elif drone_y > center_y * 5 / 8:  # 拱门在无人机下侧
                    #     print("下降")
                    #     tl_flight.down(distance=20).wait_for_completed()
                    #     stack.append(520)
                    # else:
                        print("上下位置合适")
                        anch_y = True
                        break
                else:
                    print("未找到拱门")
                time.sleep(1)
        if not anch_find:
            if j>=1:
                continue
            tl_flight.rotate(angle=20).wait_for_completed()
            stack.append(320)

        if anch_x and anch_y:
            tl_flight.rotate(angle=-1).wait_for_completed()
            tl_flight.forward(distance=450).wait_for_completed()
            stack.append(1350)
            break
    if not (anch_find and anch_x and anch_y):
        tl_flight.rotate(angle=-1).wait_for_completed()
        tl_flight.forward(distance=300).wait_for_completed()
        tl_flight.forward(distance=150).wait_for_completed()
        stack.append(1500)



def find_char():
    flag = False  # 识别到字母标志
    flag_x = False  # X坐标对齐
    flag_y = False  # Y坐标对齐
    for i in range(0, 5):
        # 无人机获取最新照片识别是否时'A'
        for j in range(0, 10):  # 连续获取10帧进行识别（1次不行)
            print("获取照片")
            img = tl_camera.read_cv2_image(strategy="newest")
            hmax, wmax = img.shape[:2]  # 取图像的长宽
            cv2.imshow("picture", img)
            cv2.waitKey(1)
            print("开始识别")
            return_value = Congnization_char('A', img)
            print("识别结束")
            flag = return_value[0]
            char_x = return_value[1][0]
            char_y = return_value[1][1]
            if flag:  # 如果识别到目标,锁定目标
                if char_x > (wmax * 5 / 8):
                    tl_flight.rotate(angle=10).wait_for_completed()
                    stack.append(310)
                    print("右转")
                elif (wmax * 3 / 8) > char_x:
                    tl_flight.rotate(angle=-10).wait_for_completed()
                    stack.append(210)
                    print("左转")
                else:
                    print("左右位置正确")
                    flag_x = True
                # if  char_y > (hmax * 5 / 8):
                #     tl_flight.down(distance=20).wait_for_completed()
                #     stack.append(420)
                #     print("下降")
                # elif char_y < (hmax * 3 / 8):
                #     tl_flight.up(distance=20).wait_for_completed()
                #     stack.append(520)
                #     print("上升")
                # else:
                    print("高低位置正确")
                    flag_y = True
        # 识别到目标且对齐时，向前并灯光显示”你好"
        if flag_x and flag_y:
            flag_x=True
            flag_y=True
            flag=True
            tl_flight.forward(distance=300).wait_for_completed()
            stack.append(1300)
            Say_hello()
            break
        print("未识别到目标，右转")
        # 未识别到目标，飞行器旋转
        if not flag:
            tl_flight.rotate(angle=-20).wait_for_completed()
            stack.append(220)
    if not (flag and flag_x and flag_y):
        print("未识别到目标降落")
        tl_flight.land().wait_for_completed()
        tl_drone.close()


def return_home():
    for i in range(0, len(stack)):
        if not stack: #如果列表空了，直接跳出
            print("列表空，跳出")
            break
        act = stack.pop()  # 获取最后一个动作
        print("本指令: {0}".format(act))
        # 指令转化
        if act < 900:
            if act // 100 == 2:  # 第一阶段的左转，对应右转
                ang = act % 200
                tl_flight.rotate(angle=ang).wait_for_completed()
                print("右转")
            elif act // 100 == 3:  # 第一阶段的右转，对应左转
                ang = act % 300
                tl_flight.rotate(angle=-ang).wait_for_completed()
                print("左转")
            elif act // 100 == 4:  # 第一阶段的上升，对应下降
                dis = act % 400
                tl_flight.down(distance=dis).wait_for_completed()
                print("下降")
            elif act // 100 == 5:  # 第一阶段的下降，对应上升
                dis = act % 500
                tl_flight.up(distance=dis).wait_for_completed()
                print("上升")
            else:
                pass
        elif act//1000==1:#第一阶段的前进,对应后退
            dis=act%1000
            tl_flight.backward(distance=dis).wait_for_completed()
            print("后退")
        elif act//1000==6:#第一阶段左移,对应右移
            dis = act % 6000
            tl_flight.left(distance=dis).wait_for_completed()
            print("右移")
        else:
            pass

def return_home1():
    tl_flight.rotate(angle=180).wait_for_completed()
    for i in range(0, len(stack)):
        if not stack: #如果列表空了，直接跳出
            print("列表空，跳出")
            break
        act = stack.pop()  # 获取最后一个动作
        print("本指令: {0}".format(act))
        # 指令转化
        if act < 900:
            if act // 100 == 2:  # 第一阶段的左转，对应右转
                ang = act % 200
                tl_flight.rotate(angle=-ang).wait_for_completed()
                print("左转")
            elif act // 100 == 3:  # 第一阶段的右转，对应左转
                ang = act % 300
                tl_flight.rotate(angle=ang).wait_for_completed()
                print("右转")
            elif act // 100 == 4:  # 第一阶段的上升，对应下降
                dis = act % 400
                tl_flight.up(distance=dis).wait_for_completed()
                print("上升")
            elif act // 100 == 5:  # 第一阶段的下降，对应上升
                dis = act % 500
                tl_flight.down(distance=dis).wait_for_completed()
                print("下降")
            else:
                pass
        elif act//1000==1:#第一阶段的前进,对应后退
            dis=act%1000
            tl_flight.forward(distance=dis).wait_for_completed()
            print("前进")
        elif act//1000==6:#第一阶段左移,对应右移
            dis = act % 6000
            tl_flight.left(distance=dis).wait_for_completed()
            print("左移")
        else:
            pass

if __name__ == '__main__':
    robomaster.config.LOCAL_IP_STR = "192.168.10.2"
    tl_drone = robot.Drone()
    tl_drone.initialize()
    tl_camera = tl_drone.camera
    tl_flight = tl_drone.flight
    tl_battery = tl_drone.battery

    battery_info = tl_battery.get_battery()
    print("Drone battery soc: {0}".format(battery_info))
    stack = []  # 栈结构存储每一次动作
    ####***************
    # *代码对应
    # *一阶段：1XXX:向前运动XXX  2XX：向左转XX  3XX：向右转XX  4XX：向上XX  5XX：向下XX 6XXX:左移XXX
    # *二阶段：1XXX:向后运动XXX  2XX：向右转XX  3XX：向左转XX  4XX：向下XX  5XX：向上XX
    ####***************
    #飞行器摄像头开启和起飞

    flight_start()
    # Across_arch()
    #识别穿越拱门
    across_anch()
    # #穿过拱门后调整位置
    tl_flight.rotate(angle=-60).wait_for_completed()
    stack.append(260)
    tl_flight.forward(distance=150).wait_for_completed()
    stack.append(1150)
    #寻找字母
    find_char()
    print(stack)
    #返回起点
    return_home()
    #tl_flight.rotate(angle=180).wait_for_completed()
    #tl_flight.forward(distance=400).wait_for_completed()
    #飞行器降落关闭
    tl_flight.land().wait_for_completed()
    tl_drone.close()
