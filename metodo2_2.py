import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import sys
import glob

# You should replace these 3 lines with the output in calibration step
DIM=(1032, 778)
K=np.array([[336.6368305835171, 0.0, 543.1909198848186], [0.0, 336.25013695181326, 377.75387002678747], [0.0, 0.0, 1.0]])
D=np.array([[-0.0004696162817495535], [-0.0037488083792449746], [-0.001191737238904899], [-1.8897810250648485e-05]])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

def undistort(img_path):    
    img = cv2.imread(img_path)

    h,w = img.shape[:2]    
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    cv2.imshow("undistorted", undistorted_img)

    cv2.imwrite('calibresult2.png', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)