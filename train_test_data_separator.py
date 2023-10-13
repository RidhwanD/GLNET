import os
import random
import shutil

dataPath = "data/NWPU-RESISC45-all"
newDataPath_test = "data/NWPU-RESISC45/test_dataset"
newDataPath_train = "data/NWPU-RESISC45/train_dataset"
label = os.listdir(dataPath)


numData = 700
test_percentage = 30


for l in label:
	path_train = newDataPath_train+"/"+l
	path_test = newDataPath_test+"/"+l
	if not os.path.exists(path_train):
		os.mkdir(path_train)
	if not os.path.exists(path_test):
		os.mkdir(path_test)


# for l in label:
# 	# Generate which images should be in the test set 
# 	test_num = int(test_percentage/100 * numData)
# 	idx_test = random.sample(range(1, numData), test_num)


# 	# Separate image into test and train folders
# 	for i in range(1,701):
# 		if(i in idx_test):
# 			shutil.copyfile(os.path.join("data/NWPU-RESISC45-all", l, l+"_"+str(i).zfill(3)+".jpg"),
#                 os.path.join("data/NWPU-RESISC45/test_dataset", l, l+"_"+str(i).zfill(3)+".jpg"))
# 		else:
# 			shutil.copyfile(os.path.join("data/NWPU-RESISC45-all", l, l+"_"+str(i).zfill(3)+".jpg"),
#                 os.path.join("data/NWPU-RESISC45/train_dataset", l, l+"_"+str(i).zfill(3)+".jpg"))