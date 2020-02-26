from matplotlib import pyplot as plt
import cv2
import numpy as np


def show_hist(img):
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])  # if channels = [0] in color image -> hist for red channel
    histg = cv2.calcHist([img], [1], None, [256], [0, 256])
    histb = cv2.calcHist([img], [2], None, [256], [0, 256])

    plt.figure("Histograma")
    plt.title("Histograma")
    plt.xlabel("Valori")
    plt.ylabel("Pixeli")
    plt.plot(histr, 'r')
    plt.plot(histg, 'g')
    plt.plot(histb, 'b')
    plt.xlim([0, 256])


def apply_mask(img_, mask):
    new_img = img_.copy()
    new_img[mask] = [100, 0, 0]

    return new_img


if __name__ == "__main__":
    img = cv2.imread("imgs/aurora.jpg")
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_img_3 = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2BGR)
    show_hist(img)
    masked_img = apply_mask(img, img[:, :, 0] < 30)
    cv2.imshow("Masked image", masked_img)

    imgs_stacked = np.vstack((img, grey_img_3))
    cv2.imshow("Stacked imgs", imgs_stacked)

    #cv2.waitKey()

    plt.show()
