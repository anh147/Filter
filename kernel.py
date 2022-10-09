#python 3

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot(data, title):
    plot.i += 1
    plt.subplot(1,2,plot.i)
    plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
    plt.gray()
    plt.title(title)
plot.i = 0

image_path = "C://Users//anhpn//Desktop//MEMS//code2//img//frame177.jpg"

image = cv2.imread(image_path)
# cv2.imshow("raw image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plot(image, 'Original Image')

# #invert image
# cv2.imshow("invert image", 255-image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#creat a 5x5 kernel low pass filter
kernel = np.ones(5, dtype=np.float32)/25
dst = cv2.filter2D(image, -1, kernel)
# cv2.imshow("image passes though LPF", dst)
# cv2.waitKey(0)
# plot(dst, "Low  Pass Filter")


#creat a 3x3 kernel high pass filter
kernel2 =  np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

dst = cv2.filter2D(image, -1, kernel2)
plot(dst, "High  Pass Filter")
# 
plt.show()