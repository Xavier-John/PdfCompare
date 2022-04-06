
import tempfile
import numpy as np
import glob
from pdf2image import convert_from_path
from compareImages import compareImage, comparePages




def pdf2Image(src):
    with tempfile.TemporaryDirectory() as tmpdirname:
     print('created temporary directory', tmpdirname)
     images = convert_from_path(src)
     images = list(map(np.array,images))
    #  tmpdirname.cleanup()
     return images

def compareFolders(masterFolder,testFolder):
    masterFileList = glob.glob(masterFolder + '/*.pdf')
    testFileList = glob.glob(testFolder + '/*.pdf')
    testFileList = np.sort(testFileList)
    masterFileList = np.sort(masterFileList)
    for masterFile,testFile in zip(masterFileList,testFileList):
        masterList = pdf2Image(masterFile)
        testList = pdf2Image(testFile)
        print(masterFile,testFile)
        x = masterFile[-7:]
        y = testFile[-7:]
        if (x != y):
            print("\nFile Mismatch")
            exit()
        if(comparePages(masterList,testList,testFile)):
            print("defect")
            with open('out.txt', 'a') as f:
                f.write(testFile)
                f.write('\n')
            f.close()

        masterList = None
        testList = None


