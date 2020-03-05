from inference.inference_general_utils import *
from inference.inference_plot_neuralnets import *
import time
import numpy as np
import pandas as pd

class VideoAnalyzer:
    """
    Implements a frame capturar from the webcam 
    """
    def __init__(self,
                 cap,
                 age_net,
                 aff_net,
                 pose_net,
                 face_net,
                 face_cascade=None,):

        self.cap = cap
        self.face_net = face_net
        self.face_analyzer = FaceAnalyzer(age_net,
                                          aff_net, 
                                          pose_net,)
        self.collected_data = []
        self.frame_nbr = 0

    def analyze_video(self):
        tic = time.time()
        fps_list = []
        while(True):
            video = self.cap.read()
            ret, frame = video
            #print(frame.shape)
            #detect faces in gray frame, so it runs faster 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.analyze_frame(frame, gray)
            cv2.imshow('frame',frame)

            tac = time.time()
            tictac = tac - tic
            tic = tac
            fps = 1/tictac
            fps_list.append(fps)
            print("Frames per second: {0}\r".format(fps), end="\r")
            self.frame_nbr += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                break

            # When everything done, release the capture

        print("\n[INFO] - Ending recording and releasing caputre from webcam")
        self.cap.release()
        cv2.destroyAllWindows()
        fps_arr = np.array(fps_list)
        print("[INFO] - FPS MEAN: ", str(np.mean(fps_arr)))
        print("[INFO] - FPS STD: ", str(np.std(fps_arr)))

    def analyze_frame(self, frame, gray):
        """
        analizes and anotates the frame
        """
        faces = self.face_net.find_faces(frame)
        for face in faces:
            ((age, gender), aff_label), (yaw, pitch, roll) = self.face_analyzer.infere_from_face(frame, gray, face)
            self.collected_data.append((self.frame_nbr, age, gender, aff_label, yaw, pitch, roll))

    def store_gathered_data(self,
                            output_path="video_data.csv"):
        """
        Stores the data from the captured frames
        """
        as_df = pd.DataFrame(self.collected_data)
        as_df.columns = ["frame", "age", "gender", "emotion", "yaw", "pitch", "roll"]
        print("\n[INFO] - saving raw collected data to csv in in path:", output_path)
        as_df.to_csv(output_path)
        return as_df 
