import os
import sys
sys.path.append("../src/")

import numpy as np
import cv2

from image_utils import *
from blur import *
from tone import *
from contrast import *
from background import *

import unittest

test_dir = "./"
input_dir = os.path.join(test_dir,"input/")
filter_dir = os.path.join(test_dir, "filter/")

image_dir = os.path.join(input_dir, "image/")
profile_dir = os.path.join(input_dir, "profile/")

blur_dir = os.path.join(filter_dir, "blur/")

def MSE(image1, image2):
    return np.sqrt(np.mean(np.square(image1 - image2)))

def IOU(image1, image2):
    src1 = (image1 >= 125)
    src2 = (image2 >= 125)

    overlap = src1 * src2
    union   = src1 + src2

    IOU = overlap.sum() / float(union.sum())
    return IOU

class TestBlurProcessing(unittest.TestCase):
    """
    영상을 Blurry하게 해주어서, Noise를 줄이는 연산
    영상 내 연산은 Machine의 Spec에 따라 연산 오류가 다르게 날 수 있으므로,
    근사값(MSE, Mean-Squared-Error)을 통해 통과 여부를 결정

    1) avgBlur     --  주위 픽셀의 평균값으로 픽셀 값을 결정하는 연산
        * test_avgBlur_ksize_11     : ksize 11만큼 Blur 연산
        * test_avgBlur_ksize_19     : ksize 19만큼 Blur 연산

    2) normBlur    --  주위 픽셀의 Gaussian 평균값으로 픽셀 값을 결정하는 연산
        * test_normBlur_ksize_11_sigma_0     : ksize 11만큼 Blur 연산
        * test_normBlur_ksize_19_sigma_0     : ksize 19만큼 Blur 연산
        * test_normBlur_ksize_19_sigma_5     : ksize 19, sigmaX 5에 대한 Blur 연산

    3) medianBlur  --  주위 픽셀의 중간값으로 픽셀 값을 결정하는 연산
        * test_medianBlur_ksize_11   : ksize 11만큼 Blur 연산
        * test_medianBlur_ksize_19   : ksize 19만큼 Blur 연산

    4) bltBlur  -- profile의 border 영역을 width만큼 추출하는 연산
        * test_bltBlur_sigma_25 : sigmaColor, sigmaSpace 25만큼 Blur 연산
        * test_bltBlur_sigma_75 : sigmaColor, sigmaSpace 75만큼 Blur 연산

    5) Blur with Mask -- Mask 연산과 Blur 연산을 조합하여 연산,
                         BackGround에만 Blurry하게 하여, out-focus 효과를 줌

        * test_normBlur_with_reversed_mask : Background에만 normBlur를 적용
    """
    def setUp(self):
        """ the testing framework will automatically
            call this for every single test we run.
        """
        test_profilename = "1_366.png" # 이 이미지로 Mask Processing test

        self.profile_dir = profile_dir
        self.image_dir = image_dir

        imagepath = os.path.join(self.image_dir,test_profilename)
        profilepath = os.path.join(self.profile_dir,test_profilename)
        self.image = read_image(imagepath)
        self.profile = read_profile(profilepath)

    def tearDown(self):
        pass

    def test_avgBlur_ksize_11(self):
        result = avgBlur(self.image, 11)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_avgBlur_ksize_11.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_avgBlur_ksize_19(self):
        result = avgBlur(self.image, 19)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_avgBlur_ksize_19.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_normBlur_ksize_11_sigma_0(self):
        result = normBlur(self.image, 11, 0)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_normBlur_ksize_11_sigma_0.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_normBlur_ksize_19_sigma_0(self):
        result = normBlur(self.image, 19, 0)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_normBlur_ksize_19_sigma_0.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_normBlur_ksize_19_sigma_5(self):
        result = normBlur(self.image, 19, 5)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_normBlur_ksize_19_sigma_5.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_medianBlur_ksize_11(self):
        result = medianBlur(self.image, 11)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_medianBlur_ksize_11.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_medianBlur_ksize_19(self):
        result = medianBlur(self.image, 19)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_medianBlur_ksize_19.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_bltBlur_sigma_25(self):
        result = bltBlur(self.image, sigmaColor=25, sigmaSpace=25)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_bltBlur_sigma_25.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_bltBlur_sigma_75(self):
        result = bltBlur(self.image, sigmaColor=75, sigmaSpace=75)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_bltBlur_sigma_75.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_normBlur_with_reversed_mask(self):
        blurred = normBlur(self.image, ksize=11)
        mask = reverse_mask(self.profile)
        result = merge_images(blurred,self.image,mask)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(blur_dir,"test_normBlur_with_reversed_mask.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

if __name__ == "__main__":
    unittest.main(verbosity=2)
