import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

a = glob.glob("./data/photo/*/*")

print(len(a)) #400


img = cv2.imread(a[0])


# Create a blank image of zeros (same dimension as img)
# It should be grayscale (1 color channel)
marker = np.zeros_like(img[:,:,0]).astype(np.int32)

# This step is manual. The goal is to find the points
# which create the result we want. I suggest using a
# tool to get the pixel coordinates.

# Dictate the background and set the markers to 1
marker[204][95] = 1
marker[240][137] = 1
marker[245][444] = 1
marker[260][427] = 1
marker[257][378] = 1
marker[217][466] = 1

# Dictate the area of interest
# I used different values for each part of the car (for visibility)
marker[235][370] = 255    # car body
marker[135][294] = 64     # rooftop
marker[190][454] = 64     # rear light
marker[167][458] = 64     # rear wing
marker[205][103] = 128    # front bumper

# rear bumper
marker[225][456] = 128
marker[224][461] = 128
marker[216][461] = 128

# front wheel
marker[225][189] = 192
marker[240][147] = 192

# rear wheel
marker[258][409] = 192
marker[257][391] = 192
marker[254][421] = 192

# Now we have set the markers, we use the watershed
# algorithm to generate a marked image
marked = cv2.watershed(img, marker)

# Plot this one. If it does what we want, proceed;
# otherwise edit your markers and repeat
plt.imshow(marked, cmap='gray')
plt.show()

# Make the background black, and what we want to keep white
marked[marked == 1] = 0
marked[marked > 1] = 255

# Use a kernel to dilate the image, to not lose any detail on the outline
# I used a kernel of 3x3 pixels
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(marked.astype(np.float32), kernel, iterations = 1)

# Plot again to check whether the dilation is according to our needs
# If not, repeat by using a smaller/bigger kernel, or more/less iterations
plt.imshow(dilation, cmap='gray')
plt.show()

# Now apply the mask we created on the initial image
final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))

# cv2.imread reads the image as BGR, but matplotlib uses RGB
# BGR to RGB so we can plot the image with accurate colors
b, g, r = cv2.split(final_img)
final_img = cv2.merge([r, g, b])

# Plot the final result
plt.imshow(final_img)
plt.show()


# # img = cv2.imread('images/watershed.jpg')
# # img = cv2.imread('images/water_coins.jpg')

# # binaray image로 변환
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# #Morphology의 opening, closing을 통해서 노이즈나 Hole제거
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

# # dilate를 통해서 확실한 Backgroud
# sure_bg = cv2.dilate(opening,kernel,iterations=3)

# #distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
# # 즉, 중심으로 부터 점점 옅어져 가는 영상.
# # 그 결과에 thresh를 이용하여 확실한 FG를 파악
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
# sure_fg = np.uint8(sure_fg)

# # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
# unknown = cv2.subtract(sure_bg, sure_fg)

# # FG에 Labelling작업
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0

# # watershed를 적용하고 경계 영역에 색지정
# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]


# images = [gray,thresh,sure_bg,  dist_transform, sure_fg, unknown, markers, img]
# titles = ['Gray','Binary','Sure BG','Distance','Sure FG','Unknow','Markers','Result']

# for i in range(len(images)):
#     plt.subplot(2,4,i+1),plt.imshow(images[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

# plt.show()