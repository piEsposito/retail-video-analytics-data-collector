from inference.inference_general_utils import *
from inference.inference_plot_neuralnets import *
import time
import numpy as np

def infer_from_video(cap, exec_age_net,exec_aff_net, exec_pose_net, face_cascade):
    tic = time.time()
    frame_nbr = 0
    collected_data = []
    fps_list = []
    while(True):
        # Capture frame-by-frame
        video = cap.read()
        ret, frame = video

        #detect faces in gray frame, so it runs faster 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #infere age, gender and expression and label the image
        for face in faces:
            ((age, gender), aff_label), (yaw, pitch, roll) = infere_from_face(frame, gray,
                                                                              face,
                                                                              exec_age_net,
                                                                              exec_aff_net,
                                                                              exec_pose_net)

            collected_data.append((frame_nbr, age, gender, aff_label, yaw, pitch, roll))

        # Display the resulting frame
        cv2.imshow('frame',frame)

        #to get the fps
        tac = time.time()
        tictac = tac - tic
        tic = tac
        fps = 1/tictac
        fps_list.append(fps)

        print("Frames per second: {0}".format(fps))
        frame_nbr += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    print("[INFO] - Ending recording and releasing caputre from webcam")
    cap.release()
    cv2.destroyAllWindows()
    fps_arr = np.array(fps_list)
    print("\n[INFO] - FPS MEAN: ", str(np.mean(fps_arr)))
    print("[INFO] - FPS STD: ", str(np.std(fps_arr)))
    return collected_data

