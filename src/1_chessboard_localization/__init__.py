import os
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":

    image_path = os.path.abspath(os.path.join(__file__, "../../../data/chessred2k/images/0/G000_IMG001.jpg"))

    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    edges = cv.Canny(img_rgb, 100, 200)

    plt.figure(figsize=(20, 10))

    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')


    plt.subplot(122) 
    plt.imshow(edges)
    plt.title("Edge Image")
    plt.axis('off')

    plt.show()
