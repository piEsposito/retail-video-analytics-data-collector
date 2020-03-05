import cv2
import argparse
from inference.inference_general_utils import *
from inference.inference_plot_neuralnets import *
from inference.infer_from_video import *
from inference.data_saver_utils import *



#here we parse the paths for the output data 
ap = argparse.ArgumentParser()
ap.add_argument("-or", "--output_raw_data_path", required=False, default="collected_data.csv",
	help="path to output raw data")
ap.add_argument("-op", "--output_preprocessed_data_path", required=False, default="collected_preprocessed_data.csv",
	help="path to output preprocessed data")

#advanced options: you can select the device you are plugin your neural nets in. 
#dont worry if you dont undersdant it, just let it be CPU
#search intel OpenVINO Inference Engine documentation to see more

ap.add_argument('-agde', '--age_net_device', required=False, default="CPU",
		help="device to plugin the OpenVINO optimized age_net neural network: MOVIDIUS, CPU, GPU")
ap.add_argument('-affde', '--aff_net_device', required=False, default="CPU",
		help="device to plugin the OpenVINO optimized aff_net neural network: MOVIDIUS, CPU, GPU")
ap.add_argument('-posede', '--pose_net_device', required=False, default="CPU",
		help="device to plugin the OpenVINO optimized pose_net neural network: MOVIDIUS, CPU, GPU")
ap.add_argument('-facede', '--face_net_device', required=False, default="CPU",
		help="device to plugin the OpenVINO optimized pose_net neural network: MOVIDIUS, CPU, GPU")

args = vars(ap.parse_args())
#print(args)
#capture video from the webcam of index 0
cap = cv2.VideoCapture(0)

#here we set up our neural networks and face classifiers 
age_net_bin = "inference/neuralnets/age-gender-recognition-retail-0013.bin"
age_net_xml = "inference/neuralnets//age-gender-recognition-retail-0013.xml"
#age_net = load_net(age_net_xml, age_net_bin, device=args["age_net_device"])
age_net = OptimizedNN(age_net_xml,
					  age_net_bin,
					  blob_shape=(62,62),
					  input_layer_name="data",
					  device=args["age_net_device"],
					  swapRB=True)

aff_net_bin = "inference/neuralnets//emotions-recognition-retail-0003.bin"
aff_net_xml = "inference/neuralnets//emotions-recognition-retail-0003.xml"
#aff_net = load_net(aff_net_xml, aff_net_bin, device=args["aff_net_device"])
aff_net = OptimizedNN(aff_net_xml,
					  aff_net_bin,
					  blob_shape=(64,64),
					  input_layer_name="data",
					  device=args["aff_net_device"],
					  swapRB=True)

pose_net_bin = "inference/neuralnets//head-pose-estimation-adas-0001.bin"
pose_net_xml = "inference/neuralnets//head-pose-estimation-adas-0001.xml"
#pose_net = load_net(pose_net_xml, pose_net_bin, device=args["pose_net_device"])
pose_net = OptimizedNN(pose_net_xml,
		 			   pose_net_bin,
		 			   blob_shape=(60,60),
		  			   input_layer_name="data",
					   device=args["pose_net_device"],
					   swapRB=True)

face_net_bin = "inference/neuralnets//face-detection-retail-0004.bin"
face_net_xml = "inference/neuralnets//face-detection-retail-0004.xml"
#face_net = load_net(face_net_xml, face_net_bin, device=args["face_net_device"])
face_net = MultiBoxNN(face_net_xml,
					   face_net_bin,
					   blob_shape=(300, 300),
					   input_layer_name="data",
					   device=args["pose_net_device"],
					   swapRB=True)

face_cascade = cv2.CascadeClassifier('inference/neuralnets/haarcascade_frontalface_default.xml')

#here we run the program
data = infer_from_video(cap, age_net, aff_net, pose_net, face_cascade, face_net)
df = store_gathered_data(data, output_path=args["output_raw_data_path"])
df_hoted = one_hot_encode_gathered_data(df, output_path=args["output_preprocessed_data_path"])
