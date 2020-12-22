# grabcut 
import numpy as np
import matplotlib.pyplot as plt
import cv2 # print(cv2.__version__) #4.4.0
import glob


a = glob.glob("./data/photo/*/*")

print(len(a)) #400



for i in range(len(a)):

    src = cv2.imread(a[i])

    # openCV: BGR, matplotlib: RGB -> 따라서 1열과 3열을 바꿔 줘야 함 
    # 출력의 문제니까 저장은 잘되지 않을까? 

    mask = np.zeros(src.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, 450, 290)
    cv2.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    src = src*mask2[:, :, np.newaxis]


    cv2.imwrite(a[i].split('.jpg')[0]+'_bg_grabcut.jpg', src, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # plt.imshow(src)
    # plt.colorbar()
    # plt.show()

