
from pdftoimage import pdf2Image
from display import plotImage
from compareImages import comparePages

import os
PATH = 'master/20220322-emr-PA.pdf'
TEST_PATH = 'test/20220322emr-PA.pdf'

def main():
    print('start')
    print(os.getcwd())
    masterList  = pdf2Image(PATH)
    testList = pdf2Image(TEST_PATH)
    comparePages(masterList,testList)

    # plotImage(masterList[0])


if __name__ == "__main__":
    main()