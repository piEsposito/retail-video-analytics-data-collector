import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork


class OptimizedNN:
    def __init__(self,
                 net_xml,
                 net_bin,
                 blob_shape,
                 input_layer_name,
                 device="CPU",
                 swapRB=True):

        self.plugin = IECore()
        self.net = IENetwork(net_xml, net_bin)
        self.exec_net = self.plugin.load_network(self.net, device)
        self.blob_shape = blob_shape
        self.swapRB = swapRB
        self.input_layer_name = input_layer_name

    def infere_from_image(self, img):
        #print(img.shape)
        blob = cv2.dnn.blobFromImage(img, 1, self.blob_shape, 0, self.swapRB)
        infers = self.exec_net.infer({self.input_layer_name:blob})
        return infers

class MultiBoxNN(OptimizedNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_faces(self, frame):
        # [1, 1, N, 7]
        out_pred = self.infere_from_image(frame)
        faces = out_pred["detection_out"]
        coords = []
        for i in range(faces.shape[2]):
            #print(i)
            face = faces[0, 0, i, :]
            if face[2] > 0.65:
                x = int(face[3] * 640)
                y = int(face[4] * 480)
                x_max = int(face[5] * 640)
                y_max = int(face[6] * 480)
                face_info = (x, y, x_max - x, y_max - y)
                #print(face_info)
                coords.append(face_info)
        return coords