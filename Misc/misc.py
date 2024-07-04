from screeninfo import get_monitors
import cv2

def adjust_res(img):
    monitors = get_monitors()
    
    for monitor in monitors:
        if monitor.is_primary:
            m_width , m_height = (monitor.width, monitor.height)
            
    img_height, img_width = img.shape[:2]

    # Calculate resize factor to fit the image within the monitor dimensions
    resize_factor = min(m_width / img_width, m_height / img_height)
    
    new_width  = int((img_width * resize_factor) * 0.9)
    new_height = int((img_height * resize_factor) * 0.9)

    return (new_width, new_height)

def print_boxes(frame, frame_detections):
    for row in range(len(frame_detections)):
            x1 = int(frame_detections[row, 0])
            y1 = int(frame_detections[row, 1])
            x2 = int(frame_detections[row, 2])
            y2 = int(frame_detections[row, 3])
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)