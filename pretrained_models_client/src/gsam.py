#!/usr/bin/env python3
import rospy
from unreal_cv_ros.msg import UeSensorRaw
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import socket
import struct
import json
import os

# list the prompt categories which are common in urban environments
prompt_texts = 'building . tree'


def show_mask(mask, image, random_color=False):
    """
    将遮罩绘制到图像上。

    mask: 二值化的遮罩矩阵
    image: 原始图像
    random_color: 是否使用随机颜色
    """
    if random_color:
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)  # 随机颜色
    else:
        # 预定义颜色 (30, 144, 255)，即 DeepSkyBlue
        color = np.array([30, 144, 255], dtype=np.uint8)

    # 创建彩色遮罩图像 (浮点类型)
    mask_image = np.zeros_like(image, dtype=np.uint8)
    mask_image[mask > 0] = color

    # 将遮罩叠加到原始图像上，设置透明度
    alpha = 0.6  # 透明度
    cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0, image)


def show_box(box, image, label=None):
    """
    在图像上绘制边框和标签。

    box: 边框，格式为 (x1, y1, x2, y2)
    image: 原始图像
    label: 要显示的标签文本
    """
    # 边框颜色 (绿色)
    color = (10, 233, 20)

    # 绘制矩形框
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

    if label is not None:
        # 设置字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # 确保文本框不超出图像范围
        text_x = max(x1, 0)  # 防止超出左边界
        text_y = max(y1 - 10, text_size[1])  # 防止文本超出上边界

        # 防止文本框超出图像右边界和底部
        image_height, image_width = image.shape[:2]
        if text_x + text_size[0] > image_width:
            text_x = image_width - text_size[0]
        if text_y > image_height:
            text_y = image_height

        # 绘制文本背景框
        cv2.rectangle(
            image,
            (text_x, text_y - text_size[1]),
            (text_x + text_size[0], text_y),
            (255, 255, 0),  # 黄色背景
            -1,
        )

        # 绘制文本
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),  # 黑色文本
            thickness,
            lineType=cv2.LINE_AA,
        )


class ImageClient:

    def __init__(self):

        self.log_path = rospy.get_param('~log_path', '/root/catkin_ws/mlogs/')
        self.bridge = CvBridge()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        connected = False
        while not connected:
            try:
                self.socket.connect(('172.31.233.148', 11991))
                connected = True
            except socket.timeout:
                print("Connection timed out. Retrying...")
            except socket.error as e:
                print(f"Socket error: {e}. Retrying...")
                # wait for 5 seconds before retrying
                rospy.sleep(5)

        self.sub = rospy.Subscriber('ue_sensor_raw', UeSensorRaw, self.callback)
        self.image_cnt = 0

    def callback(self, data):

        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data.rgb_image, "rgb8")
        except CvBridgeError as e:
            print(e)
            return

        # Encode rgb image
        _, img_encoded = cv2.imencode('.png', cv_image)
        img_bytes = img_encoded.tobytes()

        # Prepare metadata
        height, width, _ = cv_image.shape

        metadata = {
            'prompt_texts': prompt_texts,
            'image_size': len(img_bytes),
            'image_width': width,
            'image_height': height
        }

        metadata_json = json.dumps(metadata)
        metadata_size = struct.pack('!I', len(metadata_json))

        # Send size of metadata, metadata, and image
        self.socket.sendall(metadata_size)
        self.socket.sendall(metadata_json.encode())
        self.socket.sendall(img_bytes)

        # 先接收数据大小（4字节的整数）
        data_size_bytes = self.socket.recv(4)
        if len(data_size_bytes) < 4:
            raise ValueError("Failed to receive data size")

        # 解析数据大小
        data_size = struct.unpack('!I', data_size_bytes)[0]

        # 根据接收的大小来接收 JSON 数据
        mask_data_bytes = b""
        while len(mask_data_bytes) < data_size:
            packet = self.socket.recv(data_size - len(mask_data_bytes))
            if not packet:
                break
            mask_data_bytes += packet

        # 将字节解码为 JSON 格式
        mask_data = json.loads(mask_data_bytes.decode())
        masks = [np.squeeze(np.array(mask), axis=0) for mask in mask_data["masks"]]
        boxes_filt = [np.array(box) for box in mask_data['boxes']]
        pred_phrases = mask_data['labels']

        for mask in masks:
            # print the mask shape
            show_mask(mask, cv_image, random_color=True)

        # 处理每个边框和标签
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box, cv_image, label)

        # 保存图片到指定路径
        output_path = os.path.join(self.log_path, "gsam_res_{}.png".format(self.image_cnt))
        cv2.imwrite(output_path, cv_image)

        # 更新计数器
        self.image_cnt += 1

    def shutdown_callback(self):
        self.socket.close()

if __name__ == '__main__':
    rospy.init_node('gsam')
    client = ImageClient()
    rospy.on_shutdown(client.shutdown_callback)
    rospy.spin()
