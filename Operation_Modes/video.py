import cv2
import os
import logging
import numpy as np
from Misc.misc import adjust_res, print_boxes

def calculate_delay(cap):
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        print("Error: Could not retrieve frame rate.")
        exit()
    # Calculate the delay between frames in milliseconds
    delay = int(1000 / fps)
    return delay

def video_mode(args, path, recognition_model):
    file_path = os.path.join(path[3], f"{args.file}.mp4")
    cap = cv2.VideoCapture(file_path)
    frame_delay = calculate_delay(cap)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            logging.ERROR("Can't receive frame (stream end?). Exiting ...")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        new_size = adjust_res(frame)
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        
        #
        #
         ##Recognition occurs here
        Railway_ligths = recognition_model.FRSign_recognition(frame)
        #
        #
        print(Railway_ligths)
        print_boxes(frame, Railway_ligths)
               
        cv2.imshow(f"{args.file}.mp4", frame)
        
        # Wait for the calculated frame delay, exit on 'q' key press
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()