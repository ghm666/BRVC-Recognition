# 裁剪testdemo

import cv2

image = cv2.imread("test.jpg")
cv2.imshow("Original", image)
print(image.shape)
print(image)

cropImg = image[28:1988, 509:2494]
print(cropImg.shape)
cv2.imshow("crop", cropImg)
cv2.imwrite("crop.jpg",cropImg)
cv2.waitKey(0)