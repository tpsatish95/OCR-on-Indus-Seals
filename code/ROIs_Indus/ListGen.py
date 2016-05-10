import os

tr_te = "Train"

fil = open(tr_te + ".txt","w")
baseDir = "Data/" + tr_te + "/"

opt = ""
# opt = "./examples/Indus_Scripts/"
for d in os.listdir(baseDir):
	t = 0
	if d == "NoText":
		t = 0
	elif d == "Text":
		t = 1
	else:
		t = 2
	for f in os.listdir(baseDir+d):
		fil.write(opt + baseDir + d + "/" + f + " " + str(t) + "\n")

fil.close()


# ./build/tools/convert_imageset ./examples/pokemon/Train/ ./examples/pokemon/listTrain.txt ./examples/pokemon/TrainKanto100
# ./build/tools/convert_imageset ./examples/pokemon/Test/ ./examples/pokemon/listTest.txt ./examples/pokemon/TestKanto100
