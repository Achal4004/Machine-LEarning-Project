import numpy as np
import os
import cv2

iimage_dir='data'
iimage_extension='.jpg'
fimage_dir='ready_data'

#function for converting color image into grayscale imgae
def gray(img):
    return np.dot(img,[0.299,0.584,0.144])

def preProcessImage(iimage_name):

    #read/load the image
    raw_img=cv2.imread(iimage_dir+"\\"+iimage_name+iimage_extension)
    #cv2.imread function take two argument first one is path and second one is type 1--> colouredimage 0-->grayimage
    
    #convert the rgb image into gray 
    gray_img=gray(raw_img).astype('uint8')

    #apply gussian fillter for reducing the noise and imporving the smoothness of the image
    gaussian_img=cv2.GaussianBlur(gray_img,(9,9),1.75)

    #apply Contrast Limited Adaptive Histogram Equalization(CLAHE) for imporving the contrast of the image
    clahe=cv2.createCLAHE(clipLimit=5)          
    clahe_img=clahe.apply(gaussian_img)+30

    #apply gamma function for improving the brightness of the image
    gamma=1.2
    gamma_img=np.array(255*(clahe_img/255)**gamma,dtype='uint8')

    #top hat filtering
    filterSize=(128,128)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)
    tophat_img=cv2.morphologyEx(gamma_img,cv2.MORPH_TOPHAT,kernel)
    
    #make directory
    if(os.path.isdir(fimage_dir+'\\'+iimage_name)):
        pass
    else:
        os.mkdir(fimage_dir+'\\'+iimage_name)
        
    #save the updated image    
    cv2.imwrite(fimage_dir+'\\'+iimage_name+'\\'+iimage_name+"_raw.png",raw_img)
    cv2.imwrite(fimage_dir+'\\'+iimage_name+'\\'+iimage_name+"_gray.png",gray_img)
    cv2.imwrite(fimage_dir+'\\'+iimage_name+'\\'+iimage_name+"_gaussian.png",gaussian_img)
    cv2.imwrite(fimage_dir+'\\'+iimage_name+'\\'+iimage_name+"_clahe.png",clahe_img)
    cv2.imwrite(fimage_dir+'\\'+iimage_name+'\\'+iimage_name+"_gamma.png",gamma_img)
    return cv2.imwrite(fimage_dir+'\\'+iimage_name+'\\'+iimage_name+"_tophat.png",tophat_img)


print("welcome in image Pre Processing")
for i in range(1,10):
    if(preProcessImage('im000'+str(i))):
        print('image', i ,'sucessfully converted.')
    else:
        print('image',i,'conversion unsucessfull.')
for i in range(10,31):
    if(preProcessImage('im00'+str(i))):
        print('image', i ,'sucessfully converted.')
    else:
        print('image',i,'conversion unsucessfull.')

input('press a key for Exit: ')

