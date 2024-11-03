import cv2
cap = cv2.VideoCapture('dehazed_footage.mp4')
ret = 1
i=1
while ret:
    ret,frame = cap.read()
    cv2.imwrite(f"frames/frame{i}.jpg",frame)
    print(f"Written frame{i}.jpg")
    i+=1
