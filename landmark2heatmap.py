# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import os
import numpy as np
import pickle
import torch

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_landmark(img_dir, imgname):
    fulldir = img_dir + imgname
    im = cv2.imread(fulldir)
    im = cv2.resize(im, (128, 128))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) != 0:
        for (i, rect) in enumerate(rects[:1]):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(im, (x, y), 2, (0, 255, 0), -1)

        # cv2.imwrite(img_dir+'keypoints_'+name, im)
        return shape
    # cv2.imshow("Output", im)
    # cv2.waitKey()


def process_oneimg(img_dir, imgname):
    fulldir = img_dir + imgname

    img = cv2.imread(fulldir)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)



    heatmap = np.zeros([128, 128, 68])  # (of original image)
    mapofAllPoints = np.zeros([128, 128])

    keypoints = detect_landmark(img_dir, imgname)

    if keypoints is None:
        for i in range(68):
            heatmap[:, :, i] = cv2.circle(np.zeros([128, 128]), (np.random.randint(1,128),np.random.randint(1,128)), 4, 255, -1)
            cv2.circle(mapofAllPoints, (np.random.randint(1,128),np.random.randint(1,128)), 4, 255, -1)
    else:
        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            # draw circles
            if len(keypoint) != 0:
                heatmap[:, :, i] = cv2.circle(np.zeros([128, 128]), (keypoint[0], keypoint[1]), 4, 255, -1)
                cv2.circle(mapofAllPoints, (keypoint[0], keypoint[1]), 1, 255, -1)

    # cv2.imwrite(img_dir + 'heatmap_' + imgname, mapofAllPoints)

    # kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    # dilatedMapofAllPoints = cv2.dilate(mapofAllPoints, kernel, iterations=2)
    # # cv2.imwrite(img_dir + 'dilated_' + imgname, dilatedMapofAllPoints)
    # # dilatedMapofAllPoints[dilatedMapofAllPoints == 0] = 1
    # # dilatedMapofAllPoints[dilatedMapofAllPoints == 255] = 2
    #
    # g_kernel = cv2.getGaussianKernel(3, 0.5)
    # g_kernel_2D = np.outer(g_kernel, g_kernel.transpose())
    # g_heatmap = cv2.filter2D(heatmap, -1, g_kernel_2D)
    # g_dilatedMapofAllPoints = cv2.filter2D(mapofAllPoints, -1, g_kernel_2D)
    # # cv2.imwrite(img_dir + 'gaussian_filtered_' + imgname, g_dilatedMapofAllPoints)

    # return cv2.imwrite(img_dir + 'heatmap_' + imgname, mapofAllPoints)
    return mapofAllPoints

#
# if __name__ == '__main__' :
#
#     process_oneimg('C:\\Users\\Networking_Lab_009\\Desktop\\1\\', 'Rafd090_01_Caucasian_female_sad_frontal.jpg')
#

# print(detect_landmark('C:\\Users\\Networking_Lab_009\\Desktop\\1\\', 'Rafd090_01_Caucasian_female_sad_frontal.jpg'))
# process_oneimg('C:\\Users\\Networking_Lab_009\\Desktop\\1\\', 'reconst_sample.jpg')
# process_oneimg('C:\\Users\\Networking_Lab_009\\Desktop\\1\\', 'Rafd090_01_Caucasian_female_fearful_frontal.jpg')
# cv2.imwrite('C:\\Users\\Networking_Lab_009\\Desktop\\1\\' + 'heatmap_' + 'Rafd090_01_Caucasian_female_surprised_frontal.jpg', process_oneimg('C:\\Users\\Networking_Lab_009\\Desktop\\1\\', 'Rafd090_01_Caucasian_female_surprised_frontal.jpg'))

#lm = detect_landmark('C:\\Users\\Networking_Lab_009\\Desktop\\1\\', 'Rafd090_01_Caucasian_female_surprised_frontal.jpg')
# print(lm[37], lm[40], lm[41], lm[38])
# mouth_start_x = lm[49][0]-10
# mouth_finish_x = lm[55][0]+10
# mouth_start_y = max(lm[51][1], lm[52][1], lm[53][1])-5
# mouth_finish_y = lm[58][1]+5
# print(mouth_start_x, mouth_finish_x, mouth_start_y,mouth_finish_y)
#lefteye_start_x = lm[37][0] - 15
#lefteye_finish_x = lm[40][0] + 15
#lefteye_start_y = min(lm[41][1], lm[42][1]) - 20
#lefteye_finish_y = max(lm[38][1], lm[39][1]) + 10

#righteye_start_x = lm[43][0] - 15
#righteye_finish_x = lm[46][0] + 15
#righteye_start_y = min(lm[48][1], lm[47][1]) - 20
#righteye_finish_y = max(lm[44][1], lm[45][1]) + 10

#lm_image = cv2.imread('C:\\Users\\Networking_Lab_009\\Desktop\\1\\Rafd090_01_Caucasian_female_surprised_frontal.jpg')
#cv2.imwrite('C:\\Users\\Networking_Lab_009\\Desktop\\1\\test1.jpg',lm_image[righteye_start_y:righteye_finish_y,righteye_start_x:righteye_finish_x])