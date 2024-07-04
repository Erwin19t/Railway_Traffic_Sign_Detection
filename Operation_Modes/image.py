import cv2
import os
import time
from Misc.misc import adjust_res, print_boxes

def image_mode(args, path, recognition_model):
    
    start_time = time.time()
    
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
    print(Railway_ligths)

    print_boxes(img, Railway_ligths)
        
    cv2.imshow(f"{args.file}.png", img)
    
    print("Required time to detection is: %s", (time.time() - start_time))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    