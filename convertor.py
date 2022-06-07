import cv2
import numpy as np

K = 6


dice_imgs = []
for i in range(1,7):
    dimg = cv2.imread('{}.png'.format(i), 0 )
    dimg = cv2.resize( dimg, (10,10 ))
    dice_imgs.append( dimg )
    

img = cv2.imread('img1.jpg', 0 )
cv2.imshow('image', img )
img = cv2.resize( img, None, fx=0.25 , fy=0.25 )
h,w = img.shape
dh,dw = dice_imgs[0].shape


pixels = img.reshape((-1,1))
pixels = np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
components,label,center=cv2.kmeans( pixels ,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)


x = list(zip(classes, center ))
x.sort( key = lambda x:x[1])


new_label = np.zeros_like( label ) 
for i in range(len(x)):
    old_class = x[i][0]
    new_label[ label == old_class ] = i

result_img = np.zeros( (h*dh , w*dw ), dtype = np.uint8)

for i in range(h):
    for j in range(w):
        class_color = new_img[i,j]
        result_img[ i*dh : (i+1)*dh , j*dw : (j+1) * dw ] = dice_imgs[class_color] 
        

cv2.imwrite('res.png', result_img )
cv2.imshow('result_img', cv2.resize( result_img, None, fx=0.47, fy=0.47) )
cv2.waitKey(0)
