# This hack to detect eye blinks will work despite low lighting conditions.
# Histogram of gray image is computed. If the range or spread of pixels are more in 
# a histogram, then it signifies an opened eye (as an open eye will have more white 
# pixels in the cropped eye image)

import cv2
import collections 
import numpy as np
import matplotlib.pyplot as plt


pixelCount = collections.deque([]) 

runLoop = True
isEyeOpen = True

while(runLoop):

    for i in range (100):

        # print("eyeOpenClose/eye"+str(i) + ".jpg")
        # read the image
        image = cv2.imread("eyeOpenClose/eye"+str(i) + ".jpg", 1)

        # Convert to gray scale as histogram works well on 256 values.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # calculate frequency of pixels in range 0-255 
        histg = cv2.calcHist([gray],[0],None,[256],[0,256])

        # hack to know whether eye is closed or not.
        # more spread of pixels in a histogram signifies an opened eye
        activePixels = np.count_nonzero(histg)
        pixelCount.append(activePixels)

        if (len(pixelCount) > 30 and 
            activePixels > np.average(pixelCount)):

            if isEyeOpen == False:
                print(" Eyes are Open")
                isEyeOpen = True

        else:
            if isEyeOpen:
                print(" Eyes are Closed...")
                isEyeOpen = False

        cv2.namedWindow('eyeOpenClose', cv2.WINDOW_NORMAL)
        cv2.imshow('eyeOpenClose',image)
        cv2.waitKey(20)

        if (i == 99):
            i = 0

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        runLoop = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
