import numpy as np
import cv2
from time import time 

class Region:
    x = 0
    y = 0
    width = 0
    height = 0
    image = None

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def generateSubregions(self):
        return [
            Region(self.x, self.y, self.width // 2, self.height // 2),
            Region(self.x + self.width // 2, self.y, self.width // 2, self.height // 2),
            Region(self.x, self.y + self.height // 2, self.width // 2, self.height // 2),
            Region(self.x + self.width // 2, self.y + self.height // 2, self.width // 2, self.height // 2)
        ]

    def fillWithImage(self, img):
        self.image = img[self.y:self.y+self.height, self.x:self.x+self.width]    

    def getDevs(self):
        mean, stddev = cv2.meanStdDev(self.image)

        threshold = a * stddev[0]

        self.mean = mean[0]
        self.threshold = threshold 

        return self.mean, self.threshold

    def subsamplable(self):
        return self.width == 2 and self.height == 2

    def isHomogeneous(self):
        isHomogeneous = True
    
        for i in range(self.height):
            for j in range(self.width):
                if abs(self.image[i, j] - self.mean) > self.threshold:
                    self.image[i, j] = self.mean
                    isHomogeneous = False
                    break
            if not isHomogeneous:
                break
            

        return isHomogeneous


def compress(img, a, region):
    if region.width <= 1 or region.height <= 1:
        return

    region.fillWithImage(img)
    mean, threshhold = region.getDevs() 

    if threshhold / a < 1e-10:
        return

    subregions = region.generateSubregions()

    for subregion in subregions:
        compress(img, a, subregion)

    if region.subsamplable():
        img[region.y:region.y+region.height, region.x:region.x+region.width] = mean


img = cv2.imread("inputs/480px-Lenna.png", cv2.IMREAD_GRAYSCALE)

a = 75
a = 1 + a / 100

start = time()

compress(img, a, Region(0, 0, img.shape[1], img.shape[0]))

end = time()

print(end - start)

cv2.imshow("Result", img)
cv2.imwrite("results/frac_result.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



