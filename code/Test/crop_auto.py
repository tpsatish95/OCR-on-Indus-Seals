from PIL import Image, ImageChops
import cv2
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import os

def trim(im):

    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        return im.crop(bbox)

def auto_canny(image, sigma=0.33):

	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	return edged

def cropWhite(fname, dst_directory):

	threshold = 250
	while True:
		img = scipy.misc.imread(fname)
		imgf = ndimage.gaussian_filter(img, 3.0)
		labeled, nr_objects = ndimage.label(imgf > threshold)

		plt.imsave('temp.png', labeled)
		image = cv2.imread("temp.png")

		copy = image.copy()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 0)
		auto = auto_canny(blurred)

		_, cnts, _ = cv2.findContours(auto.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		screenCnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
		x,y,w,h = cv2.boundingRect(screenCnt)
		if w * h > (img.shape[0] * img.shape[1])*0.60:
			# crop_fname = fname.split(".")[0]+"_crop."+fname.split(".")[1]
			crop_fname = dst_directory + fname.split(".")[0].split("/")[-1]+"_crop.tif"
			plt.imsave(crop_fname, scipy.misc.imread(fname)[y:y+h, x:x+w])

			im = Image.open(crop_fname)
			im = trim(im)
			if im is not None:
				# im.show()
				im.save(crop_fname)
			break
		elif threshold == 200:
			crop_fname = dst_directory + fname.split(".")[0].split("/")[-1]+"_crop.tif"
			im = Image.open(fname)
			im = trim(im)
			if im is not None:
				im.show()
				im.save(crop_fname)
			break
		else:
			threshold = 200

if __name__ == "__main__":
	src_directory = "Data/Images_RMRL/"
	dst_directory = "Data/Images_RMRL_crop/"
	for file in os.listdir(src_directory):
		cropWhite(src_directory+file, dst_directory)

''' Experiments '''

# image = cv2.imread("out.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# auto = auto_canny(blurred)

# _, cnts, _ = cv2.findContours(auto.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
# screenCnt = cnts[0]
# x,y,w,h = cv2.boundingRect(screenCnt)
# cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),10)
# cv2.imshow("The Seal", copy)
# cv2.waitKey(0)


# def imResize(image, width=None, height=None, inter=cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]

#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image

#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)

#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))

#     # resize the image
#     resized = cv2.resize(image, dim, interpolation=inter)

#     # return the resized image
#     return resized

# image = cv2.imread("sample2.tif")
# ratio = image.shape[0] / 300.0
# orig = image.copy()
# image = imResize(image, height = 300)

# # convert the image to grayscale, blur it, and find edges
# # in the image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# edged = cv2.Canny(gray, 30, 200)

# _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
# screenCnt = cnts[0]

# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
# cv2.imshow("The Seal", image)
# cv2.waitKey(0)

# cv2.drawContours(image, cnts[0], -1, (0, 255, 0), 10)
# cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),10)
# cv2.imshow("The Seal", copy)
# cv2.waitKey(0)
