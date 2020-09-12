# This hack to detect eye blinks will work despite low lighting conditions.
# Histogram of gray image is computed. If the range or spread of pixels are more in 
# a histogram, then it signifies an opened eye (as an open eye will have more white 
# pixels in the cropped eye image)

import cv2
import collections 
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import peakutils.peak

pixelCount = []

runLoop = True
isEyeOpen = True


def showImage(x, peaks):
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")

    plt.show()



while(runLoop):
    print('=======================================================================')
    for i in range (540):

        # read the image
        image = cv2.imread("eyeImages_day/eye"+str(i+10) + ".jpg", 1)

        # Convert to gray scale as histogram works well on 256 values.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # calculate frequency of pixels in range 0-255 
        histg = cv2.calcHist([gray],[0],None,[256],[0,256])

        # hack to know whether eye is closed or not.
        # more spread of pixels in a histogram signifies an opened eye
        activePixels = np.count_nonzero(histg)
        pixelCount.append(activePixels)

        if len(pixelCount) > 50:

            diff = np.diff(pixelCount[-50:])
            
            peaks = peakutils.peak.indexes(np.array(diff), thres=0.8, min_dist=2)
            x =   np.array([i * -1 for i in diff])
            peaksReflected = peakutils.peak.indexes(np.array(x), thres=0.8, min_dist=2)

            # if peak is there on upright and reflected signal then the closed eyes are open soon
            # i.e. it denotes a blink and not a gesture. But if peak is found only on the reflected
            # signal then eyes are closed for long time to indicate gesture.
            if (peaksReflected.size > 0 and x[peaksReflected[0]] > 0 and peaks.size == 0):

                print('event triggered at ' + str(i+31) + '...')

                plt.plot(x)
                # plt.plot(peaks, x[peaks], "x")
                plt.plot(peaksReflected, x[peaksReflected], marker="x")
                plt.pause(10.05)

                pixelCount.clear()

        # Display the resulting frame
        cv2.imshow('frame',image)
        # print("image = " + str(i))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            runLoop = False
            exit()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
