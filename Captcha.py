import cv2
import numpy as np
import pytesseract
import sys

image = cv2.imread(sys.argv[1]) 


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((1,2),np.uint8)

opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN,kernel)

threshold_img = cv2.threshold(opening, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


#cv2.imshow('threshold image', threshold_img)
#cv2.waitKey(0)
#v2.destroyAllWindows()



text=pytesseract.image_to_string(threshold_img, config="--psm 3")
print("".join(text.split()))
