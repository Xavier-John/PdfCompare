
import tempfile
import numpy as np
import os
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

def markMissingFiles(masterFolder,testFolder):
    masterFileList = glob.glob(masterFolder + '/*.pdf')
    testFileList = glob.glob(testFolder + '/*.pdf')
    testFileList = np.sort(testFileList)
    masterFileList = np.sort(masterFileList)
    print('No of masterfiles = ',len(masterFileList))
    print('No of testFilelist =',len(testFileList))
    if (len(masterFileList)!=len(testFileList)):
        print('\nFiles are missing program may error out')
    i = 0
    for masterFile,testFile in zip(masterFileList,testFileList):
        print(masterFile,testFile)
        x = masterFile[-7:]
        y = testFile[-7:]
        z = masterFileList[i]
        z2 = testFileList[i]
        print(z,z2)
        if (x != y):
            print("\nFile Mismatch")
            
            x1 = (masterFileList[i][-7:])
            y1 = testFileList[i+1][-7:]
            x2 = (masterFileList[i+1][-7:])
            y2 = testFileList[i][-7:]
            if ((masterFileList[i][-7:])==(testFileList[i+1][-7:])):
                # rename testfile 
                
                os.rename(testFile,testFile+'_missing')
            elif ((masterFileList[i+1][-7:])==(testFileList[i][-7:])):
                # rename masterfile
                os.rename(masterFile,masterFile+'_missing')
            exit()
        i =i +1
