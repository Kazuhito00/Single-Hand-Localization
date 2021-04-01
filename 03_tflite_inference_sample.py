#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy

import cv2 as cv
import numpy as np
import tensorflow as tf

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument(
        "--model",
        # default='02_model/MobileNetV1/SingleHandLocalization_1.0_128.tflite')
        default='02_model/EfficientNetB0/SingleHandLocalization_224.tflite')
    # parser.add_argument("--input_shape", type=int, default=128)
    parser.add_argument("--input_shape", type=int, default=224)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_path = args.model
    input_shape = args.input_shape

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    interpreter = Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    IMAGE_WIDTH = input_shape
    IMAGE_HEIGHT = input_shape

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)  # ミラー表示
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 検出実施 #############################################################
        x = cv.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = x.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        x = x.astype('float32')
        x /= 255

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        predict_result00 = interpreter.get_tensor(output_details[0]['index'])
        predict_result01 = interpreter.get_tensor(output_details[1]['index'])

        class_id = np.argmax(np.squeeze(predict_result00))
        class_score = np.squeeze(predict_result00)[class_id]
        point = np.squeeze(predict_result01)
        point_x = int(point[0] * frame_width)
        point_y = int(point[1] * frame_height)

        if class_id == 0 and class_score < 0.5:
            pass
        elif class_id == 1:
            cv.circle(debug_image, (point_x, point_y), 12, (255, 0, 0), -1)
        elif class_id == 2:
            cv.circle(debug_image, (point_x, point_y), 12, (0, 255, 0), -1)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 処理時間計測 ########################################################
        elapsed_time = time.time() - start_time

        cv.putText(
            debug_image,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5,
            cv.LINE_AA)
        cv.putText(
            debug_image,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv.LINE_AA)

        # 画面反映 #############################################################
        cv.imshow('tflite inference sample', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
