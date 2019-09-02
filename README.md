# retail-video-analytics-data-collector

**What it is?**
The retail-video-analytics-data-collector is a Python script that captures frames from your webcam, identifies any faces on it. Run each faces through several neural networks optimized with OpenVINO Model Optimizer gathers the age, gender, emotion (infered from facial expression) and head pose. It stores the data in two csv files: one with the raw data and other with scaled and one-hot encoded data, in the case you want to do some machine learning with it. My objective was to crate a project using Intel-Powered technology, with its pretrained models and OpenVINO toolkit. 

**How it works?** 
 In a few words, it loads the optimized and trained neuralnets and plugs in into a device, loads the face classifier. We then capture each frame, identify the faces on it with the face classifier, get the features for each face with the neuralnets and store it. We got the stored data, copy it, preprocess it with standartscaling and one-hot encoding and, same, store it. In the inference folder we also have a Jupyter Notebook, so you can see each block of code working and its paper on this little program.
 
 To run it, once the environment is set, you just need to:
 
 
 `git clone https://github.com/piEsposito/retail-video-analytics-data-collector.git`
 
 
 Go to the root project directory and then:
 
 
 `python main.py`
 
 
 If you know how it works, you can select the device on which your are plugging in your neural networks in the commandline with:
 
 `--age_net_device`
 
 
 `--aff_net_device`
 
 
 `--pose_net_device`
 
 
 (or run it with -h to see the command line options)
 
 
 If you have any more doubts, the functions are all commented. Don't hesitate in contacting me if you have any feedback, issue, want to criticize, have any suggestion or just talk :) And I would be so happy if anyone pull requests me here :)
 
 **What do I need to make it work?**
You must have python 3.6 installed and OpenVINO installed. I am developing it using Intel Distribution for Python, so I do recommend you use it. I am using Ubuntu and, pretty sure, it does not work on Windows yet (due to "os.sep" stuff)

To install, you run: 


`conda update conda`


`conda config --add channels intel`


`conda create -n idp intelpython3_full python=3`


`source activate idp`

You also should install OpenVINO:

To install it, you can follow this links instructions:

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#install-openvino
https://software.intel.com/en-us/openvino-toolkit/choose-download

BEFORE YOU RUN IT, DONT FORGET TO SETUP OPENVINO ENVIRONMENT VARIABLES WITH THIS LINE OF CODE:


`source /opt/intel/openvino/bin/setupvars.sh`

**And I don't have to download any neural-networks or packages other than the above ones?**


No, you dont. All you need to run it is in this repo, including the optimized neural-networks and its labels. I know that is very frustrating to see a unreproducible project on github due to absence of the trained models and its labels, so I did put it all here. 

