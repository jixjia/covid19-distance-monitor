import cv2

def transparent_box(image, bbox, color=(0,250,0), alpha=0.5):
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    
    # draw a red rectangle surrounding Adrian in the image
    # along with the text "PyImageSearch" at the top-left
    # corner
    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
    
    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output
