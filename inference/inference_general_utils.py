import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork

def load_net(net_xml, net_bin, device="CPU"):
    '''
    this function gets a neural net intermediate representation paths for both xml and bin,
    turns it into a IENetwork from OpenVINO and plugs it into a device
    '''
    plugin = IECore()
    net = IENetwork(net_xml, net_bin)

    exec_net = plugin.load_network(net, device)
    return exec_net

def infere_from_image(img, blob_shape, exec_net, swapRB=True, input_layer_name="data"):
    '''
    this functions run a matrix-representation of a imege (or a frame), resizes it so it can run troghout the network
    and gets its outputs as the selected layers
    blob_shape = (n, m) as the input shape of the neural network
    '''
    #img = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(img, 1, blob_shape, 0,swapRB)
    infers = exec_net.infer({input_layer_name:blob})
    return infers
