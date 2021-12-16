import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0


for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis=1)

while (cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    image = np.flip(image, axis=1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bg = cv2.resize(bg,(640,480))
    image = cv2.resize(image,(640,480))

    upper_black = np.array([104, 153, 70])
    lower_black = np.array([30, 30,0])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    res = cv2.bitwise_and(bg, bg, mask=mask)

    f = bg - res 
    f = np.where(f == 0,image , f)
    
   
    cv2.imshow("video", bg) 
    cv2.imshow("mask", f) 
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

    cv2.imshow("magic", f)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()