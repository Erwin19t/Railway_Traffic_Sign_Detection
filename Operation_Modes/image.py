import cv2
import os
import time
from Misc.misc import adjust_res, print_boxes
from Algorithm import Blink_Detection

def image_mode(args, path, recognition_model):    
    file_path = os.path.join(path[2], f"{args.file}.png")
    img = cv2.imread(file_path)
    
    new_size = adjust_res(img)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    
    #
    #
    ##Recognition occurs here
    Railway_ligths = recognition_model.FRSign_recognition(img)
    #
    #
    Blink_Detection(img, 1, Railway_ligths)
    #print(Railway_ligths)
    #print_boxes(img, Railway_ligths)
        
    cv2.imshow(f"{args.file}.png", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    