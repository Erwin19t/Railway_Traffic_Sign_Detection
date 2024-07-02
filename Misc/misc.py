from screeninfo import get_monitors

def adjust_res(img):
    monitors = get_monitors()
    
    for monitor in monitors:
        if monitor.is_primary:
            m_width , m_height = (monitor.width, monitor.height)
            
    img_height, img_width = img.shape[:2]

    # Calculate resize factor to fit the image within the monitor dimensions
    resize_factor = min(m_width / img_width, m_height / img_height)
    
    new_width  = int(img_width * resize_factor) 
    new_height = int(img_height * resize_factor)

    return (new_width, new_height)