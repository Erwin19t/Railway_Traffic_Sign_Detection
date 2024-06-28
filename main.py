import argparse
import logging

def parse_arguments():
    #Initializes an instance of the 'ArgumentParser' class from the 'argparse' module
    parser = argparse.ArgumentParser()
    
    #Add hyperparameters as arguments
    parser.add_argument("-m", "--mode", type=int, choices=[0, 1], help="0 for image & 1 for video")
    parser.add_argument("-f", "--file", type=str, help="Type the filename")
    
     #Parse command line arguments
    return parser.parse_args()

def path_list():
    #List of paths required in the program
    return (
        "./YOLOV5/",                                            # 0: Model Path
        "./YOLOV5/runs/train/Plate/weights/Weights.pt",         # 1: License Weights Path
        "./YOLOV5/runs/train/Characters/weights/Weights.pt",    # 2: Characters Weights Path
        "./Tests/Images",                                # 3: Image mode operation Path
        "./Tests/Videos",                                # 4: Video mode operation Path
    )

def main(args, path):
    #Initializes an instance of the 'RecognitionModel' class
    recognition_model = None
    if args.mode:
        logging.info("Video mode has been chosen...")
    else:
        logging.info("Image mode has been chosen...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    path = path_list()
    main(args, path)
