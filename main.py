import argparse
import logging
import os
from Operation_Modes import image_mode, video_mode
from Model.model import RecognitionModel

def parse_arguments():
    #Initializes an instance of the 'ArgumentParser' class from the 'argparse' module
    parser = argparse.ArgumentParser()
    
    #Add hyperparameters as arguments
    parser.add_argument("-m", "--mode", type=str, default="Image", help="Available modes: Image & Video")
    parser.add_argument("-f", "--file", type=str, default="001", help="Type the filename")
    parser.add_argument("-e", "--exp", type=str, default="Erwin_exp_00", help="Check the Experiments folder & type the one of your preference")
    
     #Parse command line arguments
    return parser.parse_args()

def path_list():
    # List of paths required in the program
    base_path = "."
    return (
        os.path.join(base_path, "yolov5"),            # 0: Model Path
        os.path.join(base_path, "Experiments"),       # 1: Experiments Path
        os.path.join(base_path, "Tests", "Images"),   # 2: Image Test Folder
        os.path.join(base_path, "Tests", "Videos"),   # 3: Video Test Folder
    )
    
def main(args, path):
    #Initializes an instance of the 'RecognitionModel' class
    recognition_model = RecognitionModel(path, args)
    if args.mode == "Video":
        logging.info(" Video mode has been chosen...")
        video_mode(args, path, recognition_model)
        return
    
    elif args.mode == "Image":
        logging.info(" Image mode has been chosen...")
        image_mode(args, path, recognition_model)
        return
    
    else:
        logging.info(" Not available mode")

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    args = parse_arguments()
    path = path_list()
    main(args, path)
