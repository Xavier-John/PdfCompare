
from pdftoimage import pdf2Image,compareFolders,markMissingFiles
from display import plotImage
from compareImages import comparePages
import argparse



import os
PATH = 'master/20220324-emr-AR.pdf'
TEST_PATH = 'test/20220324emr-AR.pdf'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-P','--path',default='wdt25')
    arg = ap.parse_args()
    path = arg.path
    print('start')
    print(os.getcwd())
    markMissingFiles('master/'+path,'test/'+path)
    compareFolders('master/'+path,'test/'+path)
    # masterList  = pdf2Image(PATH)
    # testList = pdf2Image(TEST_PATH)
    # comparePages(masterList,testList)

    # plotImage(masterList[0])


if __name__ == "__main__":
    main()
