# -*- coding: utf-8 -*-
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import skimage.transform
import os
import shutil
import caffe
from PIL import Image

candidates = set()
merged_candidates = set()
refined = set()
final = set()
final_extended = set()
text_boxes = set()
text=set()
text_cut = set()
no_text = set()
text_cut_final = set()

def getClass(FileList):
    caffe.set_mode_gpu()
    classifier = caffe.Classifier("../ROIs_Indus/deploy.prototxt","../ROIs_Indus/Models/bvlc_googlenet_indusnet_iter_20000.caffemodel" ,
            image_dims=[224,224], raw_scale=255.0, channel_swap = [2,1,0])

    inputs = [caffe.io.load_image(im_f) for im_f in FileList]
    print("Classifying %d inputs." % len(inputs))

    predictions = classifier.predict(inputs)

    return predictions

def texbox_ext():
    global text
    global both_text
    global text_cut_final
    for x, y, w, h in text:
        A = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h, 'w': w, 'h': h}
        for x1, y1, w1, h1 in both_text:
            B = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}

            # overlap between A and B
            SA = A['w']*A['h']
            SB = B['w']*B['h']
            SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
            SU = SA + SB - SI
            overlap_AB = float(SI) / float(SU)

            overf = 0
            ax1,ay1,aw,ah = A['x1'],A['y1'],A['w'],A['h']
            if overlap_AB > 0.0:
                if A['x1'] > B['x1'] and abs(B['x1']+B['w'] - A['x1']) < A['w']*0.20: # B is left to A
                    ax1 = B['x1']
                    aw =  A['x1'] + A['w'] - B['x1']
                    overf = 1
                # if A['y1'] < B['y1'] and abs(A['y1']-B['y1']) > A['h']*0.70: # B is bottom to A
                #     ah = A['h'] - (A['y1']+A['h'] - B['y1'])
                #     overf = 1
                # if A['y1'] > B['y1']: # B is top to A
                #     ay1 = B['y1'] + B['h']
                if A['x1'] < B['x1']: # B is right to A
                    aw = B['x1']+B['w'] - A['x1']
                    overf = 1
                # if A['y1'] < B['y1']: # B is bottom to A
                #     ah = A['h'] - (A['y1']+A['h'] - B['y1'])
                # REPLACE by Cohen Suderland algo

                A['x1'],A['y1'],A['w'],A['h'] = ax1,ay1,aw,ah
                text_cut_final.add((A['x1'],A['y1'],A['w'],A['h']))
            if overf == 1:
                break
        text_cut_final.add((A['x1'],A['y1'],A['w'],A['h']))
    text_cut_final = text_cut_final - both_text # CHANGE THIS LINE

def texbox_cut():
    global no_text
    no_text = no_text.union(both_text)

    for x, y, w, h in text:
        A = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h, 'w': w, 'h': h}
        for x1, y1, w1, h1 in no_text:
            B = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}

            # overlap between A and B
            SA = A['w']*A['h']
            SB = B['w']*B['h']
            SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
            SU = SA + SB - SI
            overlap_AB = float(SI) / float(SU)

            overf = 0
            ax1,ay1,aw,ah = A['x1'],A['y1'],A['w'],A['h']
            if overlap_AB > 0.0:
                if A['x1'] > B['x1'] and abs(B['x1']+B['w'] - A['x1']) < A['w']*0.20: # B is left to A
                    ax1 = B['x1'] + B['w']
                    overf = 1
                if A['y1'] < B['y1'] and abs(A['y1']-B['y1']) > A['h']*0.70: # B is bottom to A
                    ah = A['h'] - (A['y1']+A['h'] - B['y1'])
                    overf = 1
                # if A['y1'] > B['y1']: # B is top to A
                #     ay1 = B['y1'] + B['h']
                # if A['x1'] < B['x1']: # B is right to A
                #     aw = A['w'] - (A['x1']+A['w'] - B['x1'])
                # if A['y1'] < B['y1']: # B is bottom to A
                #     ah = A['h'] - (A['y1']+A['h'] - B['y1'])
                # REPLACE by Cohen Suderland algo

                A['x1'],A['y1'],A['w'],A['h'] = ax1,ay1,aw,ah
                text_cut.add((A['x1'],A['y1'],A['w'],A['h']))
            if overf == 1:
                break
        text_cut.add((A['x1'],A['y1'],A['w'],A['h']))


def extend_text_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[3] for i in l]))

def draw_textbox():
    global width, height
    thresh = ((width+height)/2)*(0.25)
    tempc = set()
    for x, y, w, h in text_boxes:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        f = 0
        for x1, y1, w1, h1 in text_boxes:
            if abs(y1-y) <= thresh and abs(h1-h) <= thresh:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
                f = 1
        if f == 0:
            text.add((x, y, w, h))
        text.add(extend_text_rect(temp))

def contains():
    x1, y1, w1, h1 = p
    for x, y, w, h in candidates:
        if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
            return True
        if x1<=x and y1 <= y and x1+w1 >= x+w and y1+h1 >= y+h:
            candidates.remove((x, y, w, h))
            return False
    return False

def extend_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[3] for i in l]))

def extend_superbox():
    global width, height
    thresh = ((width+height)/2)*(0.06)

    tempc = set()
    for x, y, w, h in final:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        for x1, y1, w1, h1 in final:
            if abs(y1-y) <= thresh and abs(h1-h) <= thresh:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        final_extended.add(extend_rect(temp))


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

            if overlap_A >= 0.40 or overlap_B >= 0.40:
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

def contains_remove():

    for x, y, w, h in merged_candidates:
        f = False
        temp = set(merged_candidates)
        temp.remove((x, y, w, h))
        for x1, y1, w1, h1 in temp:
            if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
                f = False
                break
            # if x1<=x and y1 <= y and x1+w1 >= x+w and y1+h1 >= y+h:
            else:
                f = True
        if f == True:
            refined.add((x, y, w, h))

# def contains_remove():
#     for x, y, w, h in merged_candidates:
#         temp = set(merged_candidates)
#         temp.remove((x, y, w, h))
#         test = []
#         for x1, y1, w1, h1 in temp:
#             A = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h, 'w': w, 'h': h}
#             B = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}
#             # overlap between A and B
#             SA = A['w']*A['h']
#             SB = B['w']*B['h']
#             SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
#             SU = SA + SB - SI
#             overlap_AB = float(SI) / float(SU)
#             if overlap_AB > 0.0:
#                 # if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
#                 if x1<=x and y1 <= y and x1+w1 >= x+w and y1+h1 >= y+h:
#                     test.append(False)
#                 else:
#                     test.append(True)
#             else:
#                 test.append(True)
#         if all(test):
#             refined.add((x, y, w, h))

def mean_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[1]+i[3] for i in l]) - min([i[1] for i in l]))

def merge():
    global width, height
    thresh = int(((width+height)/2)*(0.14))
    tempc = set()
    for x, y, w, h in candidates:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        for x1, y1, w1, h1 in candidates:
            if abs(x1-x) <= thresh and abs(y1-y) <= thresh and abs(w1-w) <= thresh and abs(h1-h) <= thresh:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        merged_candidates.add(mean_rect(temp))
    contains_remove()

for name in os.listdir("./Images"):
	candidates = set()
	merged_candidates = set()
	refined = set()
	final = set()
	final_extended = set()
	text_boxes = set()
	text=set()
	text_cut = set()
	no_text = set()

	print("Processing Image " + name.split(".")[0])
	fname = "./Images/" + name
	print(fname)
	img = skimage.io.imread(fname)
	width = len(img[0])
	height = len(img)
	# new_size = 256
	# height = int(new_size * height / width)
	# width  = new_size

	if width*height < 256*256*(0.95) and abs(width-height) <= 3 :
	    new_size  = 512
	    height = int(new_size * height / width)
	    width  = new_size
	    print("A")
	elif width*height < 220*220*(1.11):
	    new_size  = 256
	    height = int(new_size * height / width)
	    width  = new_size
	    print("B")
	elif width*height < 256*256:
	    new_size  = 256
	    height = int(new_size * height / width)
	    width  = new_size
	    print("B1")
	elif width*height > 512*512*(0.99) and width < 800 and height < 800:
	    new_size  = 512
	    height = int(new_size * height / width)
	    width  = new_size
	    print("C")
	elif width*height < 512*512*(0.95) and width*height > 256*256*(1.15):
	    new_size  = 512
	    height = int(new_size * height / width)
	    width  = new_size
	    print("D")
	tried = []
	while True:
		tried.append(width)
		candidates = set()
		merged_candidates = set()
		refined = set()
		final = set()
		final_extended = set()
		text_boxes = set()
		text=set()
		text_cut = set()
		no_text = set()
		stage = 1
		text_cut_final = set()

		for sc in [350,450,500]:
			for sig in [0.8]:
				for mins in [30,60,120]: # important
					img = skimage.io.imread(fname)[:,:,:3]
					if height == len(img) and width == len(img[0]):
						pass
					else:
						img = skimage.transform.resize(img, (height, width))

					img_lbl, regions = selectivesearch.selective_search(
					img, scale=sc, sigma= sig,min_size = mins)

					for r in regions:
						# excluding same rectangle (with different segments)
						if r['rect'] in candidates:
							continue
						# excluding regions smaller than 2000 pixels
						if r['size'] < 2000:
							continue
						# distorted rects
						x, y, w, h = r['rect']
						if w / h > 1.2 or h / w > 1.2:
							continue
						if w >= (img.shape[0]-1)*(0.7) and h >= (img.shape[1]-1)*(0.7):
							continue
						candidates.add(r['rect'])
			print("Stage " + str(stage) + " Complete.")
			stage+=1

		print(candidates)
		merge()
		print(refined)
		draw_superbox()
		print(final)
		extend_superbox()
		print(final_extended)

		os.makedirs("Regions/"+name.split(".")[0])

		# draw rectangles on the original image
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(img)
		for x, y, w, h in final_extended:
			rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
			ax.add_patch(rect)
		plt.savefig("Regions/"+name.split(".")[0]+"/FinalRegions.png")
		plt.close('all')

		img1 = skimage.io.imread(fname)[:,:,:3]
		if height == len(img1) and width == len(img1[0]): pass
		else: img1 = skimage.transform.resize(img1, (height, width))
		# imgT = Image.open(fname).convert('L')
		# w, h = imgT.size
		# if height == h and width == w:
		# 	pass
		# else:
		# 	# img1 = skimage.transform.resize(img1, (height, width))
		# 	imgT = imgT.resize((width,height), Image.ANTIALIAS)

		ij = 1
		fList = []
		box_list = []
		for x, y, w, h in final_extended:
			skimage.io.imsave("Regions/"+name.split(".")[0]+"/"+str(ij)+"_sub.jpg", img1[y:y+h,x:x+w])
			# imgT.crop((x,y,x+w,y+h)).save("Regions/"+name.split(".")[0]+"/"+str(ij)+"_sub_b.png")
			# imgT = Image.open("Regions/"+name.split(".")[0]+"/"+str(ij)+"_sub.png").convert('L')
			# imgT.save("Regions/"+name.split(".")[0]+"/"+str(ij)+"_sub_b.png")
			fList.append("Regions/"+name.split(".")[0]+"/"+str(ij)+"_sub.jpg")
			box_list.append((x, y, w, h))
			ij+=1

		# classify text no text
		text_boxes=set()
		text = set()
		no_text = set()
		both_text = set()
		text_cut_final = set()

		i = 0
		try:
			a = getClass(fList)
			l = np.array([0,1,2])
			for pred in a:
				idx = list((-pred).argsort())
				pred = l[np.array(idx)]
				if pred[0] == 1 or pred[0] == 2:
					text_boxes.add(box_list[i])
				elif pred[0] == 0:
					no_text.add(box_list[i])
				if pred[0] == 2:
					both_text.add(box_list[i])
				print(pred)
				i+=1
		except:
			print("No Text Regions")

		draw_textbox()
		print(text)
		texbox_cut()
		print(text_cut)
		texbox_ext()
		print(text_cut_final)

		# draw rectangles on the original image
		img = skimage.io.imread(fname)[:,:,:3]
		if height == len(img) and width == len(img[0]): pass
		else: img = skimage.transform.resize(img, (height, width))

		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(img)
		for x, y, w, h in text_cut_final:
			rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
			ax.add_patch(rect)

		plt.savefig("Result/final_"+name.split(".")[0]+".png")
		plt.close('all')

		ij = 1
		for x, y, w, h in text_cut_final:
			skimage.io.imsave("Regions/"+name.split(".")[0]+"/"+str(ij)+"_text.png", img[y:y+h,x:x+w])
			ij+=1

		# min area check
		minf = 0
		for x, y, w, h in text_cut_final:
			if w*h < width*height*0.20  and (w < width*0.20 or h < height*0.20):
				minf = 1

		if (len(text_cut_final) == 0 or minf == 1) and len(tried) < 3:
			print(tried)
			print("New size being tried.")
			shutil.rmtree("Regions/"+name.split(".")[0]+"/")
			img = skimage.io.imread(fname)
			twidth = len(img[0])
			theight = len(img)

			new_size = list(set([256,512,twidth]) - set(tried))[0]
			height = int(new_size * theight / twidth)
			width = new_size

		else:
			break
