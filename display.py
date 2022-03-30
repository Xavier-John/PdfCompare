import cv2
import matplotlib.pyplot as plt

def displayImage(image,name,size = 1):
    '''
    Display the image using imshshow
    Params: Name,
    Image numpy array
    Size divider
    '''
    dheight = image.shape[0]
    dwidth = image.shape[1]
    dheight = dheight/size
    dwidth = dwidth/size
    showImg = cv2.resize(image,(int(dwidth),int(dheight)))
    # showImg = cv2.resize(image,(1920,1080))
    # cv2.imshow(name,showImg)


def plotImage(image):
    '''
    https://stackoverflow.com/questions/43228246/show-grayscale-opencv-image-with-matplotlib
    '''
     
#   plt.imshow(image)
    # x= len(image.shape)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),cmap='gray')
    else:
        plt.imshow(image,cmap='gray')
    plt.show()