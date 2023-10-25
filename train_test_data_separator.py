import os
import random
import shutil

# Change accordingly
dataPath = "data\WHU-RS19-all"
newDataPath_test = "data\WHU-RS19\\test_dataset"
newDataPath_train = "data\WHU-RS19\\train_dataset"
label = os.listdir(dataPath)
fill = 2
ext = ".jpg"

# Assumption: files is in the format 'FolderName_XX.ext' or 'FolderName-XX.ext'
# For consistency, rename to 'foldername_XX.ext'

# numData = 700
test_percentage = 30
    
def countFiles(dir_path):
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            os.rename(os.path.join(dir_path, path), os.path.join(dir_path, path.lower()))
            os.rename(os.path.join(dir_path, path), os.path.join(dir_path, path.replace('-','_')))
            count += 1
    return count

for l in label:
	path_train = os.path.join(newDataPath_train, l)
	path_test = os.path.join(newDataPath_test, l)
	if not os.path.exists(path_train):
		os.mkdir(path_train)
	if not os.path.exists(path_test):
		os.mkdir(path_test)


for l in label:
	# Generate which images should be in the test set 
	numData = countFiles(os.path.join(dataPath, l))
	test_num = int(test_percentage/100 * numData)
	idx_test = random.sample(range(1, numData), test_num)


	# Separate image into test and train folders
	for i in range(1,numData+1):
		if(i in idx_test):
			shutil.copyfile(os.path.join(dataPath, l, l.lower()+"_"+str(i).zfill(fill)+ext),
                os.path.join(newDataPath_test, l, l.lower()+"_"+str(i).zfill(fill)+ext))
		else:
			shutil.copyfile(os.path.join(dataPath, l, l.lower()+"_"+str(i).zfill(fill)+ext),
                os.path.join(newDataPath_train, l, l.lower()+"_"+str(i).zfill(fill)+ext))