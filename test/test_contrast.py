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
lut_dir = os.path.join(input_dir,"lut/")

contrast_dir = os.path.join(filter_dir,"contrast/")

def MSE(image1, image2):
    return np.sqrt(np.mean(np.square(image1 - image2)))

def IOU(image1, image2):
    src1 = (image1 >= 125)
    src2 = (image2 >= 125)

    overlap = src1 * src2
    union   = src1 + src2

    IOU = overlap.sum() / float(union.sum())
    return IOU

class TestContrastProcessing(unittest.TestCase):
    """
    영상의 대비를 강조하여, 몸의 형태가 강조되도록 하는 연산
    영상 내 연산은 Machine의 Spec에 따라 연산 오류가 다르게 날 수 있으므로,
    근사값(MSE, Mean-Squared-Error)을 통해 통과 여부를 결정

    1) claheContrast   -- contrast limited adaptive histogram equalization
        * test_claheContrast_with_clip_2_grid_11    : clipSize=2, gridSize=11로 연산
        * test_claheContrast_with_clip_2_grid_19    : clipSize=2, gridSize=19로 연산
        * test_claheContrast_with_clip_4_grid_11    : clipSize=4, gridSize=11로 연산
        * test_claheContrast_with_clip_4_grid_19    : clipSize=4, gridSize=19로 연산

    2) sharpening      -- 영상의 경계선을 부각시킴
        * test_sharpening_with_sharpness_03    : sharpness=0.3으로 연산
        * test_sharpening_with_sharpness_10    : sharpness=1.0으로 연산
        * test_sharpening_with_sharpness_15    : sharpness=1.5으로 연산

    3) gammaContrast   -- 주어진 Look-Up-Table을 이용하여 변경
        * test_gammaContrast_with_g_08   : gamma=0.8으로 연산
        * test_gammaContrast_with_g_10   : gamma=1.0으로 연산
        * test_gammaContrast_with_g_12   : gamma=1.2으로 연산

    4) autogammaContrast -- 사진 내 밝기값을 고려하여 GammaCorrection을 적용
        * test_autoGammaContrast_with_60    : thr=60  으로 연산
        * test_autoGammaContrast_with_80    : thr=80  으로 연산
        * test_autoGammaContrast_with_100   : thr=100 으로 연산

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

    def test_claheContrast_with_clip_2_grid_11(self):
        result = claheContrast(self.image,2,11)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_claheContrast_with_clip_2_grid_11.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_claheContrast_with_clip_2_grid_19(self):
        result = claheContrast(self.image,2,19)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_claheContrast_with_clip_2_grid_19.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_claheContrast_with_clip_4_grid_11(self):
        result = claheContrast(self.image,4,11)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_claheContrast_with_clip_4_grid_11.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_claheContrast_with_clip_4_grid_19(self):
        result = claheContrast(self.image,4,19)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_claheContrast_with_clip_4_grid_19.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_sharpening_with_sharpness_03(self):
        result = sharpening(self.image, 0.3)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_sharpening_with_sharpness_03.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_sharpening_with_sharpness_10(self):
        result = sharpening(self.image, 1.)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_sharpening_with_sharpness_10.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_sharpening_with_sharpness_15(self):
        result = sharpening(self.image, 1.5)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_sharpening_with_sharpness_15.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_gammaContrast_with_g_08(self):
        result = gammaContrast(self.image,0.8)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_gammaContrast_with_g_08.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_gammaContrast_with_g_10(self):
        result = gammaContrast(self.image,1.)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_gammaContrast_with_g_10.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_gammaContrast_with_g_12(self):
        result = gammaContrast(self.image, 1.2)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_gammaContrast_with_g_12.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_autoGammaContrast_with_60(self):
        result = autoGammaContrast(self.image,self.profile,60)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_gammaContrast_with_g_12.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_autoGammaContrast_with_80(self):
        result = autoGammaContrast(self.image,self.profile,80)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_gammaContrast_with_g_12.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_autoGammaContrast_with_100(self):
        result = autoGammaContrast(self.image,self.profile,100)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(contrast_dir,"test_gammaContrast_with_g_12.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

if __name__ == "__main__":
    unittest.main(verbosity=2)
