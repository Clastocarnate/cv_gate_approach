import cv2
cap = cv2.VideoCapture('Assets_Adaptive/adapt_thresh_enhanced.mp4')
ret = 1
i=1
while ret:
    ret,frame = cap.read()
    cv2.imwrite(f"Assets_Adaptive/frames/frame{i}.jpg",frame)
    print(f"Written frame{i}.jpg")
    i+=1
