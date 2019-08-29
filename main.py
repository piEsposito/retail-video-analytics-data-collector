import cv2
from inference.inference_general_utils import *
from inference.inference_plot_neuralnets import *
from inference.infer_from_video import *
from inference.data_saver_utils import *


cap = cv2.VideoCapture(0)

age_net_bin = "inference/neuralnets/age-gender-recognition-retail-0013.bin"
age_net_xml = "inference/neuralnets//age-gender-recognition-retail-0013.xml"
exec_age_net = load_net(age_net_xml, age_net_bin, device="CPU")

aff_net_bin = "inference/neuralnets//emotions-recognition-retail-0003.bin"
aff_net_xml = "inference/neuralnets//emotions-recognition-retail-0003.xml"
exec_aff_net = load_net(aff_net_xml, aff_net_bin, device="CPU")

pose_net_bin = "inference/neuralnets//head-pose-estimation-adas-0001.bin"
pose_net_xml = "inference/neuralnets//head-pose-estimation-adas-0001.xml"
exec_pose_net = load_net(pose_net_xml, pose_net_bin, device="CPU")

face_cascade = cv2.CascadeClassifier('inference/neuralnets/haarcascade_frontalface_default.xml')

data = infer_from_video(cap, exec_age_net,exec_aff_net, exec_pose_net, face_cascade)
df = store_gathered_data(data, output_path="collected_data.csv")
df_hoted = one_hot_encode_gathered_data(df)
