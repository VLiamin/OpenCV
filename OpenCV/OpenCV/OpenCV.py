import numpy as np
import cv2 as cv
import imutils
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt


class Tasks_OpenCV():
    
    def loading_displaying_saving():
        img = cv.imread('C:\spbu\picture.jpg')
        cv.imshow('Picture', img)
        cv.waitKey(0)

    def find_orb(self):
        img = cv.imread('C:\spbu\picture.jpg')
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

        cv.imshow('Find orb', img2)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_sift(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        # Applying SIFT detector
        sift = cv.SIFT_create()
        kp = sift.detect(img, None)

        # Marking the keypoint on the image using circles
        img2=cv.drawKeypoints(
            img,
            kp,
            img,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv.imshow('Find sift', img2)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_canny(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        edges = cv.Canny(img, 25, 255, L2gradient=False)

        cv.imshow('Canny edges', edges)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def convert_grayscale(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cv.imshow('Gray image', gray)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def convert_hsv(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        cv.imshow('Convert hsv', img_hsv)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def draw_right(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        img_right = np.fliplr(img)
        
        cv.imshow('Draw right', img_right)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def draw_botton(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        img_botton = cv.flip(img, 0)
        
        cv.imshow('Draw botton', img_botton)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def rotate_image45(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')
        rotated = imutils.rotate(img, angle=45)
        
        cv.imshow('Rotate 45', rotated)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def rotate_image_around_back_point(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        (h, w) = img.shape[:2]

        M = cv.getRotationMatrix2D((h, w), 30, 1.0)
        rotated = cv.warpAffine(img, M, (w, h))
        
        cv.imshow('Rotate around back point', rotated)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def replace_image(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')
        h, w = img.shape[:2]
        translation_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
        dst = cv.warpAffine(img, translation_matrix, (w, h))
        
        cv.imshow('Replace image', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def change_brightness(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')
        hsvImg = cv.add(img,np.array([50.0]))

        cv.imshow('Change brightness', hsvImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def change_contrast(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')
        alpha = 1.5 # Contrast control (1.0-3.0)

        adjusted = cv.convertScaleAbs(img, alpha=alpha)

        cv.imshow('Change contrast', adjusted)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @classmethod
    def gamma_conversion(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')
        gamma = 0.5

        adjusted = self.gamma_сorrection(img, gamma=gamma)

        cv.imshow('Gamma converion', adjusted)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def gamma_сorrection(src, gamma):
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv.LUT(src, table)

    def histogram_equalization(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        dst = cv.equalizeHist(src)

        cv.imshow('Histogram equalization', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @classmethod
    def make_cool(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        increaseLookupTable = self.spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv.split(img)
        red_channel = cv.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        cool = cv.merge((red_channel, green_channel, blue_channel))

        cv.imshow('Make warm', cool)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @classmethod
    def make_warm(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        increaseLookupTable = self.spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel  = cv.split(img)
        red_channel = cv.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        warm = cv.merge((red_channel, green_channel, blue_channel))

        cv.imshow('Make cool', warm)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def spreadLookupTable(x, y):
      spline = UnivariateSpline(x, y)
      return spline(range(256))

    def change_palette(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        im_color = cv.applyColorMap(img, cv.COLORMAP_BONE)


        cv.imshow('Change palette', im_color)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_image_binarization(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        median = np.median(img)
        lower = int(max(0, (1.0 - 0.33) * median))
        upper = int(min(255, (1.0 + 0.33) * median))

        edge_image= cv.Canny(img, lower, upper)

        cv.imshow('Make binarization', edge_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_conturs(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        median = np.median(img)
        lower = int(max(0, (1.0 - 0.33) * median))
        upper = int(min(255, (1.0 + 0.33) * median))

        edge_image= cv.Canny(img, lower, upper)

        contours, hierarchy = cv.findContours(edge_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #create an empty image for contours
        img_contours = np.zeros(img.shape)
        # draw the contours on the empty image
        cv.drawContours(img_contours, contours, -1, (0,255,0), 3)

        cv.imshow('Find conturs', img_contours)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_sobel_filter(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        sobel=cv.Sobel(img,cv.CV_64F,0,1,ksize=3)

        cv.imshow('Sobel filter', sobel)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_laplacian_filter(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        s = cv.Laplacian(img, cv.CV_16S, ksize=3)

        plt.imshow(s), plt.show()

    def make_blurry_image(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        s = cv.GaussianBlur(img, ksize=(9, 9), sigmaX=0, sigmaY=0)

        cv.imshow('Make blurry', s)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_low_frequencies(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        cv.imshow('Make low frequencies', magnitude_spectrum)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_fast_frequencies(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        rows, cols = img.shape[:2]
        crow,ccol = rows//2 , cols//2
        fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)

        cv.imshow('Make fast frequencies', img_back)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_erosion(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(img, kernel, iterations = 1)

        cv.imshow('Make erosion', erosion)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def make_dilate(self):
        # Loading the image
        img = cv.imread('C:\spbu\picture.jpg')

        kernel = np.ones((5, 5), 'uint8')

        dilate_img = cv.dilate(img, kernel, iterations=1)

        cv.imshow('Make dilate', dilate_img)
        cv.waitKey(0)
        cv.destroyAllWindows()





