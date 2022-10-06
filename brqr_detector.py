#
# author:      L. Saetta
# last update: 06/10/2022
#
import torch
import cv2
import numpy as np


class BarcodeQrcodeDetector:
    # class 0 is barcode
    # class 1 is qrcode
    PARENT_MODEL = "ultralytics/yolov5"
    # if you want to show the images with barcodes detectd (for example, in a NB)
    SHOW = False
    # if we want to apply yolov5 TTA in inference (default = True), can be changed
    # without TTA is a little bit faster
    AUGMENT = True

    def _load_model(self):
        self.model = torch.hub.load(self.PARENT_MODEL, "custom", path=self.model_path)
        # and set the confidence level
        self.model.conf = self.confidence

    def __init__(self, model_path, confidence=0.25):
        self.model_path = model_path
        self.confidence = confidence

        # here we load the model. Requires Internet connection to the PyTorch hub
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

    # this is a utility method.... not to be called from outside
    def do_crop_for_class(self, results, class_name):
        # class could be barcode, qrcode barcode o qrcode
        # results is what returned from model()

        imgs = results.crop(save=False)

        list_imgs_barcode = []

        for img in imgs:
            if class_name.lower() in img["label"]:
                im = img["im"]
                # trasforma l'immagine in RGB altrimenti icolori sono cambiati
                img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                list_imgs_barcode.append(img_rgb)

        return list_imgs_barcode

    # the function take as input the img read as np array and returns a matrix
    # with one row for every barcode/qrcode detected and
    # (xmin, ymin, xmax, ymax, confidence, class#, class)
    # This one return BB rectangles
    def detect_1d2d_codes(self, img: np.ndarray):
        # using TTA
        results = self.model(img, augment=self.AUGMENT)

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

    # the function take as input the img read as np array and returns a list
    # with the images cropped (numpy array, RGB, H,W,C)
    # This one return images + class_name
    def detect_and_crop_1d2d_codes(self, img: np.ndarray):
        # using TTA
        results = self.model(img, augment=self.AUGMENT)

        # first the barcodes and then qrcodes
        list_barcodes = self.do_crop_for_class(results, "barcode")
        list_qrcodes = self.do_crop_for_class(results, "qrcode")

        # create a list of (img, class_name)
        result_list = []

        if len(list_barcodes) > 0:
            for im in list_barcodes:
                result_list.append((im, "barcode"))

        if len(list_qrcodes) > 0:
            for im in list_qrcodes:
                result_list.append((im, "qrcode"))

        return result_list
