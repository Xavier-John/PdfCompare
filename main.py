
from pdftoimage import pdf2Image,compareFolders
from display import plotImage
from compareImages import comparePages

import os
PATH = 'master/20220324-emr-AR.pdf'
TEST_PATH = 'test/20220324emr-AR.pdf'

def main():
    print('start')
    print(os.getcwd())
    compareFolders('master/wdt25','test/wdt25')
    # masterList  = pdf2Image(PATH)
    # testList = pdf2Image(TEST_PATH)
    # comparePages(masterList,testList)

    # plotImage(masterList[0])


if __name__ == "__main__":
    main()
