import cv2
import os

cam = cv2.VideoCapture(1)

count = 0

label = input('Enter label of image :')

IMAGE_PATH = os.path.join('original-images')
if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

dirImage = os.path.join(IMAGE_PATH, label)
if not os.path.exists(dirImage):
    os.mkdir(dirImage)

while 1:
    _, frame = cam.read()
    cv2.imshow('Frame', frame)

    if count > 20:
        break

    if cv2.waitKey(1) == 32:
        imgName = os.path.join(IMAGE_PATH, label, label + '-' + '{}.jpg'.format(str(count)))
        cv2.imwrite(imgName, frame)
        count += 1

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()