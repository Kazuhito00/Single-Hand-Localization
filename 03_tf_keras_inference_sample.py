#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy

import cv2 as cv
import numpy as np
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        # default='02_model/MobileNetV1/SingleHandLocalization_1.0_128.hdf5')
        default='02_model/EfficientNetB0/SingleHandLocalization_224.hdf5')
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
    loaded_model = tf.keras.models.load_model(model_path)
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

        predict_result = loaded_model.predict(x)

        class_id = np.argmax(np.squeeze(predict_result[1]))
        class_score = np.squeeze(predict_result[1])[class_id]
        point = np.squeeze(predict_result[0])
        point_x = int(point[0] * frame_width)
        point_y = int(point[1] * frame_height)

        if class_id == 0 and class_score < 0.5:
            pass
        elif class_id == 1:
            cv.circle(debug_image, (point_x, point_y), 10, (255, 0, 0), -1)
        elif class_id == 2:
            cv.circle(debug_image, (point_x, point_y), 10, (0, 255, 0), -1)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 処理時間計測 ########################################################
        elapsed_time = time.time() - start_time

        cv.putText(
            debug_image,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        # 画面反映 #############################################################
        cv.imshow('tf keras inference sample', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
