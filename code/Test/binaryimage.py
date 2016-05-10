import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian_filter
import skimage.io
import skimage.color
import skimage.morphology
import numpy as np
import os

candidates = set()
refined = set()
final = set()
final_extended = set()
ref = set()

def extend_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[1]+i[3] for i in l]) - min([i[1] for i in l]))

def extend_superbox():
    global width, height
    thresh = ((width+height)/2)*(0.22)

    tempc = set()
    for x, y, w, h in final:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        for x1, y1, w1, h1 in final:
            # if abs(x1-x) <= thresh and abs(w1-w) <= thresh:
            if x1 >= x and (w1+x1) <= w+x:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        final_extended.add(extend_rect(temp))
    contains_remove(final_extended)

def draw_superbox(finals=[]):
    noover = []
    refinedT = []

    global final
    final = set()

    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    if finals != []:
        refinedT = finals
    else:
        refinedT = refined
    remp = set(refinedT)
    ref = list(refinedT)

    while len(ref) > 0:
        x1, y1, w1, h1 = ref[0]

        if len(ref) == 1: # final box
            final.add((x1, y1, w1, h1))
            ref.remove((x1, y1, w1, h1))
            remp.remove((x1, y1, w1, h1))
        else:
            ref.remove((x1, y1, w1, h1))
            remp.remove((x1, y1, w1, h1))

        over = set()
        for x2, y2, w2, h2 in remp:
            A = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}
            B = {'x1': x2, 'y1': y2, 'x2': x2+w2, 'y2': y2+h2, 'w': w2, 'h': h2}

            # overlap between A and B
            SA = A['w']*A['h']
            SB = B['w']*B['h']
            SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
            SU = SA + SB - SI
            overlap_AB = float(SI) / float(SU)
            overlap_A = float(SI) / float(SA)
            overlap_B = float(SI) / float(SB)
            # print(overlap_AB)
            #

            if overlap_A >= 0.15 or overlap_B >= 0.15:
                over.add((B['x1'],B['y1'],B['w'],B['h']))
        # print(len(over))
        if len(over) != 0: #Overlap
            remp = remp - over
            for i in over: ref.remove(i)
            over.add((A['x1'],A['y1'],A['w'],A['h']))
            # print(over)
            final.add((min([i[0] for i in over]), min([i[1] for i in over]), max([i[0]+i[2] for i in over]) - min([i[0] for i in over]), max([i[1]+i[3] for i in over]) - min([i[1] for i in over])))
            # final.add((np.mean([i[0] for i in over]), np.mean([i[1] for i in over]), np.mean([i[2] for i in over]), np.mean([i[3] for i in over])))
            noover.append(False)
        else:   #No overlap
            final.add((x1,y1,w1,h1))
            noover.append(True)

    if all(noover):
        return
    else:
        draw_superbox(final)
        return

def contains_remove(param = []):
	if param != []:
		can = set(param)
	else:
		can = set(candidates)
	for x, y, w, h in can:
		temp = set(can)
		temp.remove((x, y, w, h))
		test = []
		for x1, y1, w1, h1 in temp:
			A = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h, 'w': w, 'h': h}
			B = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}
		    # overlap between A and B
			SA = A['w']*A['h']
			SB = B['w']*B['h']
			SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
			SU = SA + SB - SI
			overlap_AB = float(SI) / float(SU)
			if overlap_AB > 0.0:
			    # if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
			    if x1<=x and y1 <= y and x1+w1 >= x+w and y1+h1 >= y+h:
			        test.append(False)
			    else:
			        test.append(True)
			else:
				test.append(True)
		if all(test) and param == []:
		    refined.add((x, y, w, h))
		if all(test) and param != []:
			ref.add((x, y, w, h))

for d in os.listdir("Regions"):
    for name in os.listdir("Regions/"+d):
        print(d)
        if "_text" in name:
            candidates = set()
            refined = set()
            final = set()
            final_extended = set()
            ref = set()
            image = skimage.io.imread("Regions/"+d+"/"+name)
            image = skimage.color.rgb2gray(image)
            thresh = threshold_otsu(image)
            binary = image <= thresh

            skimage.io.imsave("temp/binary_"+d+".png", binary)
            bin = skimage.io.imread("temp/binary_"+d+".png")
            im = gaussian_filter(bin, sigma=3.5)
            blobs = im > im.mean()

            labels = skimage.morphology.label(blobs, neighbors = 4)


            blobs = ndimage.find_objects(labels)

            plt.imsave("temp/blobs_"+d+".png", im)

            image1 = skimage.io.imread("Regions/"+d+"/"+name)
            width = len(image1[0])
            height = len(image1)
            for c1, c2 in blobs:
            	if (c2.stop - c2.start) * c1.stop - c1.start > (image1.shape[0]*image1.shape[1])*(0.026):
            		if (c2.stop - c2.start) * c1.stop - c1.start < (image1.shape[0]*image1.shape[1])*(0.90):
            			candidates.add((c2.start, c1.start,c2.stop - c2.start, c1.stop - c1.start))

            print(candidates)
            contains_remove()
            print(refined)
            draw_superbox()
            print(final)
            extend_superbox()
            print(final_extended)
            print(ref)

            image1 = skimage.io.imread("Regions/"+d+"/"+name)
            # draw rectangles on the original image
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.imshow(image1)
            for x, y, w, h in ref:
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
            plt.savefig("Regions/"+d+"/"+"patch_"+d+".png")
            plt.savefig("patches/"+"patch_"+d+".png")
            plt.close('all')


###MISC
# from skimage.viewer import ImageViewer
# viewer = ImageViewer(blobs)
# viewer.show()
