import torch
import os
import numpy as np
from typing import Union, Optional, List


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None): #costruttore per instanziare l'oggetto YOLO
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception( #controlla il device, se usare cuda o cpu fino a riga 15
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load model, carico il modello yolov5 dentro la classe YOLO "this"
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        except Exception as e:
            raise Exception("Failed to load model from {}".format(model_path))

    def __call__( #override di una funzione di default di python per istanziare un oggetto
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections
