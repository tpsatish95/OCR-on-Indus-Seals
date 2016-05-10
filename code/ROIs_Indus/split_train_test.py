import os
import random
import shutil
from PIL import Image

folders = sorted(os.listdir("Data_Raw"))

split = 0.3

for f in folders:
	os.makedirs("Data/Train/"+f)
	os.makedirs("Data/Test/"+f)
	files = sorted(os.listdir("Data_Raw/"+f))
	random.shuffle(files)

	s = int(len(files)*(1-split))+1

	for i in files[:s]:
		# shutil.copy("Data_Raw/"+f+"/"+i,"Data/Train/"+f+"/"+i)
		img = Image.open("Data_Raw/"+f+"/"+i).convert('RGB')
		img.save("Data/Train/" + f + "/" + i.split(".")[0] + '.jpg')
	for i in files[s:]:
		# shutil.copy("Data_Raw/"+f+"/"+i,"Data/Test/"+f+"/"+i)
		img = Image.open("Data_Raw/"+f+"/"+i).convert('RGB')
		img.save("Data/Test/" + f + "/" + i.split(".")[0] + '.jpg')
