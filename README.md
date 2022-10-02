# Barcode and QRcode Detector
This repository contains all the work I have done to develop a **barcode/qrcode** detector using **YOLOv5**

## Setup
To use the code for the barcode/qrcode detector you need to download the file with the trained model, in a local directory.

You can get the file from [here](https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frqap2zhtzbe/b/barcode_models/o/best_barcode_data6_yolov5x_100ep.pt).

Then, when you instantiate the Detector class, you have to specify the pathname for the pt file.

## The model.
The model to detect Bounding Box Rectangles for barcode/qrcode has been developed using [Ultralytics](https://github.com/ultralytics/yolov5) implementation of YOLO V5.

The model has been trained over around 110 jpg images, for 100 epochs.

These are the **performance metrics** measured on the validation set:
    
|Class     |Images  |Instances      |P          |R       |mAP50   |mAP50-95 |
|----------|--------|---------------|-----------|--------|--------|---------|
|   all    |   21   |      163      |   0.966   |  0.962 |  0.979 |   0.78  |
| barcode  |   21   |      143      |   0.945   |  0.923 |  0.963 |   0.712 |
| qrcode   |   21   |      20       |   0.987   |  1     |  0.995 |   0.849 |


