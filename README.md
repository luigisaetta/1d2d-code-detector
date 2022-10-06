[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Barcode and QRcode Detector
This repository contains all the work I have done to develop a **barcode/qrcode** detector using **YOLOv5**

The **model** detects Bounding Box Rectangles for barcode/qrcode. It has been developed using [Ultralytics](https://github.com/ultralytics/yolov5) implementation of **YOLO V5**.

![barcodes detected](https://github.com/luigisaetta/1d2d-code-detector/blob/main/screenshot.jpg "barcodes")

## Setup
To use the code for the barcode/qrcode detector you need to download the file with the trained PyTorch model, in a local directory.

You can get the file from [here](https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frqap2zhtzbe/b/barcode_models/o/best_barcode_data6_yolov5x_100ep.pt).

Then, when you instantiate the Detector class, you have to specify the pathname for the pt file.

See, as an example, this [NB](https://github.com/luigisaetta/1d2d-code-detector/blob/main/test_qrcode_detector.ipynb).

## Usage
If you want to get the BB rectangles:
```
from brqr_detector import BarcodeQrcodeDetector

# instantiate
detector = BarcodeQrcodeDetector(MODEL_BARCODE_PATH, CONFIDENCE)

# read the image
img1 = read_image("./img1.jpg")

# get a vector with all BB, confidence and classes
boxes = detector.detect_1d2d_codes(img1)

```
If you want to get br/qr cropped:
```
from brqr_detector import BarcodeQrcodeDetector

# instantiate
detector = BarcodeQrcodeDetector(MODEL_BARCODE_PATH, CONFIDENCE)

# read the image
img1 = read_image("./img1.jpg")

# get a list of pairs (img, class_name)
imgs_and_classes = detector.detect_and_crop_1d2d_codes(img1)

```

## The model.
The model has been trained using a 1GPU **V100**, in Oracle Data Science.

The model has been trained over around **110 jpg** images, for 100 epochs.

These are the **performance metrics**, measured on the validation set:
    
|Class     |Images  |Instances      |P          |R       |mAP50   |mAP50-95 |
|----------|--------|---------------|-----------|--------|--------|---------|
|   all    |   21   |      163      |   0.966   |  0.962 |  0.979 |   0.78  |
| barcode  |   21   |      143      |   0.945   |  0.923 |  0.963 |   0.712 |
| qrcode   |   21   |      20       |   0.987   |  1     |  0.995 |   0.849 |

## Dependencies
Obviously, you don't need to download the code for YOLO V5 to use the Detector.

The only dependencies are:
* **Torch**, version 1.10.0 or above
* **CV2**, version 4.6.0 or above

The versions listed here are the versions I have used for dev/test.

