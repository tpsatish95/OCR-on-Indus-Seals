import os
import random
import shutil
from PIL import Image

folders = sorted(os.listdir("Data"))

split = 0.20

for f in folders:
	os.makedirs("Data_U_Full/Train/"+f)
	os.makedirs("Data_U_Full/Test/"+f)
	files = sorted(os.listdir("Data/"+f))
	random.shuffle(files)

	s = int(len(files)*(1-split))+1

	for i in files[:s]:
		if ".gif" not in i:
			# shutil.copy("Data_Raw/"+f+"/"+i,"Data/Train/"+f+"/"+i)
			img = Image.open("Data/"+f+"/"+i)
			img.save("Data_U_Full/Train/" + f + "/" + i.split(".")[0].replace(" ","-") + '.jpg')
	for i in files[s:]:
		if ".gif" not in i:
			# shutil.copy("Data_Raw/"+f+"/"+i,"Data/Test/"+f+"/"+i)
			img = Image.open("Data/"+f+"/"+i)
			img.save("Data_U_Full/Test/" + f + "/" + i.split(".")[0].replace(" ","-") + '.jpg')
