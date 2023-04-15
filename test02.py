# 可以使用 cv2 去处理 rs 读到的信息
import pyrealsense2 as rs
import cv2
import numpy as np
import torch
from mydeepsort import MyDeepSort
from mydetect import MyDetect
from utils.general import xywh2xyxy
# from utils.plots import Annotator

import random
import pyrealsense2 as rs
# realsense 的参数设置
pipeline = rs.pipeline() 
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

depth_stream = pipeline.start(config)

model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model.conf = 0.5

def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)

def dectshow(org_img, boxs,depth_data, xywhs, confs, clss):
    
    img = org_img.copy()
    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, img)
    if len(outputs):
        # for j, (output, conf), box in enumerate(zip(outputs, confs)), boxs:
        for j, (output, conf, box) in enumerate(zip(outputs, confs, boxs)):
            bboxes = output[0:4]
            id = output[4]
            cls = output[5]
            c = int(cls)
            dist = get_mid_pos(org_img, box, depth_data, 24)
            label = f'{id}{det.names[c]}{str(dist / 1000)[:4]} m {conf:.2f}'
            plot_one_box(bboxes, img, label=label, color=det.colors[c], line_thickness=2)
    cv2.imshow('dec_img', img)

    # for box in boxs:
    #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    #     dist = get_mid_pos(org_img, box, depth_data, 24)
    #     cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
    #                 (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.imshow('dec_img', img)

def _():
    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, color_image)
    if len(outputs):
            
        for j, (output, conf) in enumerate(zip(outputs, confs)):
            bboxes = output[0:4]
            id = output[4]
            cls = output[5]
            c = int(cls)
            label = f'{id}{det.names[c]}{conf:.2f}'
            plot_one_box(bboxes, color_image, label=label, color=det.colors[c], line_thickness=2)


if __name__ == '__main__':
    det = MyDetect()
    deepsort = MyDeepSort()
    # cap = cv2.VideoCapture(4)   #打开默认相机

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # 在此处将 image 传递给 yolo5 进行目标检测 

        # 3. 将深度图像数据转换为深度值数组
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_array = np.asanyarray(depth_frame.get_data())
        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        depth = depth_scale * depth_array.astype(float)

        # _, image = cap.read()
        # if image is None:
        #     break
        xywhs, confs, clss = det.detect(color_image)

        results = model(color_image)
        boxs= results.pandas().xyxy[0].values
        #boxs = np.load('temp.npy',allow_pickle=True)
        
        if xywhs is None or min(xywhs.shape) == 0 or min(confs.shape) == 0 or min(clss.shape) == 0:
            pass
        else:
            dectshow(color_image, boxs, depth_image, xywhs, confs, clss)
            # print(type(xywhs.shape[0]), xywhs.shape[0])
            # outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, color_image)
            # if len(outputs):
                
            #     for j, (output, conf) in enumerate(zip(outputs, confs)):
            #         bboxes = output[0:4]
            #         id = output[4]
            #         cls = output[5]
            #         c = int(cls)
            #         label = f'{id}{det.names[c]}{conf:.2f}'
            #         plot_one_box(bboxes, color_image, label=label, color=det.colors[c], line_thickness=2)
        # cv2.imshow("test", color_image)   #显示图像
        if cv2.waitKey(1) == ord('q'):  #
            break

    pipeline.stop()
    cv2.destroyAllWindows()