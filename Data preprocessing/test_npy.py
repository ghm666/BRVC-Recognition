import cv2
import numpy as np

g = np.load("F:\\doctor_labeled_3class\\brvo2\\brvo1.npy")
# g = np.load("BROV1.npy")
print(g.shape)
cv2.imshow("Green", g)
cv2.waitKey(0)
cv2.destroyAllWindows()