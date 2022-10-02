import torch
import cv2
import numpy as np


class BarcodeQrcodeDetector:
    # class 0 is barcode
    # class 1 is qrcode
    PARENT_MODEL = "ultralytics/yolov5"
    # if you want to show the images with barcodes detectd (for example, in a NB)
    SHOW = False

    def _load_model(self):
        self.model = torch.hub.load(self.PARENT_MODEL, "custom", path=self.model_path)
        # and set the confidence level
        self.model.conf = self.confidence

    def __init__(self, model_path, confidence=0.25):
        self.model_path = model_path
        self.confidence = confidence

        self._load_model()

    # reduce the # of decimal digits
    def _reduce_digits(self, vet):
        # for every row
        for row in vet:
            # take the first four columns (xmin, ymin...)
            for i in range(4):
                # remove dec digits
                row[i] = round(row[i], 1)

            # and the fifth (confidence level)
            row[4] = round(row[4], 3)

        return vet

    # the function take as input the img read as np array and returns a matrix
    # with one row for every barcode/qrcode detected and
    # (xmin, ymin, xmax, ymax, confidence, class#, class)
    def detect_1d2d_codes(self, img: np.ndarray):
        # using TTA
        results = self.model(img, augment=True)

        if self.SHOW:
            results.print()
            print(results.pandas().xyxy[0])
            results.show()

        # some processing to remove some decimal digits
        vet_boxes = results.pandas().xyxy[0].values

        if vet_boxes.shape[0] > 0:
            vet_boxes = self._reduce_digits(vet_boxes)

        # this way we return a array with one row for detected barcode
        # (xmin, ymin, xmax, ymax, confidence, class#, class)

        # consider that barcodes are returned in order of decreasing confidence, not by location
        # but you have the BB coords... so you can establish which is above and which is below
        return vet_boxes
