import cv2
import os
from Misc.misc import adjust_res

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
    print(Railway_ligths)

    for row in range(len(Railway_ligths)):
        x1 = int(Railway_ligths[row, 0])
        y1 = int(Railway_ligths[row, 1])
        x2 = int(Railway_ligths[row, 2])
        y2 = int(Railway_ligths[row, 3])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    cv2.imshow(f"{args.file}.png", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()