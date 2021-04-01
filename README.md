# Single-Hand-Localization
単一の手の位置推定を行う試験的なモデルです。<br>
手のXY座標とクラス分類結果(無し、パー、グー)の出力を行います。<br><br>
<img src="https://user-images.githubusercontent.com/37477845/113296671-bfc6ee80-9334-11eb-8c39-231a8daff3c0.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/113308151-db37f680-9340-11eb-9f30-8efda9a85948.gif" width="45%"><br>
左図：PC実行 EfficientNet版 ONNXモデル<br>
右図：Raspberry pi4実行 MobileNet v1版 TensorFlow-Liteモデル<br><br>
MobileNet v1版はUnity Barracuda上でも動作します。興味のある方は[Unity-Barracuda-SingleHandLocalization-WebGL](https://github.com/Kazuhito00/Unity-Barracuda-SingleHandLocalization-WebGL)を確認ください。

# Requrement
* mediapipe 0.8.1 or later ※01_create_dataset.pyを利用する場合のみ
* Tensorflow 2.4.0 or later
* OpenCV 3.4.2 or Later
* onnxruntime 1.5.2 or later ※03_onnx_inference_sample.py

# Demo
Webカメラを使ったデモの実行方法は以下です。<br>
TensorFlow Kerasモデル、TensorFlow-Liteモデル、ONNXモデルそれぞれのサンプルがあります。<br>
背景が無地のほうが検出しやすいです。
```bash
python 03_tf_keras_inference_sample.py
```
```bash
python 03_tflite_inference_sample.py
```
```bash
python 03_onnx_inference_sample.py
```

デモ実行時には、以下のオプションが指定可能です。
* --device<br>カメラデバイス番号の指定 (デフォルト：0)
* --width<br>カメラキャプチャ時の横幅 (デフォルト：960)
* --height<br>カメラキャプチャ時の縦幅 (デフォルト：540)
* --model<br>ロード対象のモデル <br>03_tf_keras_inference_sample.py：'02_model/EfficientNetB0/SingleHandLocalization_224.hdf5'<br>03_tflite_inference_sample.py：'02_model/EfficientNetB0/SingleHandLocalization_224.tflite'<br>03_onnx_inference_sample.py：'02_model/EfficientNetB0/SingleHandLocalization_224.onnx'
* --input_shape<br>モデル入力画像の一辺のサイズ (デフォルト：224)

# Directory
<pre>
│  01_create_dataset.py
│  02_train_model(MobileNetV1).ipynb
│  03_tf_keras_inference_sample.py
│  03_tflite_inference_sample.py
│  03_onnx_inference_sample.py
│  
├─01_dataset
│  
├─02_model
│  ├─EfficientNetB0
│  │  │  SingleHandLocalization_224.hdf5
│  │  │  SingleHandLocalization_224.onnx
│  │  └─ SingleHandLocalization_224.tflite
│  │          
│  └─MobileNetV1
│      │  SingleHandLocalization_1.0_128.hdf5
│      │  SingleHandLocalization_1.0_128.onnx
│      └─ SingleHandLocalization_1.0_128.tflite
│          
└─utils
    └─cvfpscalc.py
</pre>
<details>
<summary>ディレクトリ内容</summary>
### 01_create_dataset.py
データセット作成用のスクリプトです。<br>
データセットは01_datasetディレクトリに保存します。

### 02_train_model(MobileNetV1).ipynb
モデル訓練用のスクリプトです。<br>
Google Colaboratory上での実行を想定しています。

### 03_tf_keras_inference_sample.py, 03_tflite_inference_sample.py, 03_onnx_inference_sample.py
推論サンプルです。

### 01_dataset
01_create_dataset.pyによって作成したデータセットを格納するディレクトリです。<br>
顔出しNGの方のデータがあるため、今回モデル訓練に使用したデータセットは非公開です。

### 02_model
訓練済みのモデルを格納しているディレクトリです。<br>
EfficientNet-B0ベース(入力サイズ224*224)とMobileNet v1ベース(128*128)のモデルを格納しています。

### utils
FPS計測用のモジュールを格納しています。
</details>

# Dataset
顔出しNGの方の写真も利用しているため、データセットは非公開です。<br>
今回訓練に使用したデータ総数は以下の通りです。<br>
　総数：65405枚<br>
　クラスID 0(手無し)：7824<br>
　クラスID 1(パー)：28605<br>
　クラスID 2(グー)：28976<br>
<br>
mediapipeを利用したデータセットを収集するスクリプト(01_create_dataset.py)は公開しています。<br>
<img src="https://user-images.githubusercontent.com/37477845/113300676-1fbf9400-9339-11eb-8377-79a1d74c4785.gif" width="50%">

# Model
モデル構造は以下の通りです。<br>
<img src="https://user-images.githubusercontent.com/37477845/113304028-98741f80-933c-11eb-8307-083c89358b83.png" width="70%">

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)

# License 
Single-Hand-Localization is under [Apache v2 License](LICENSE).
