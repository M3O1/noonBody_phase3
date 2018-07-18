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

tone_dir = os.path.join(filter_dir, "tone/")

def MSE(image1, image2):
    return np.sqrt(np.mean(np.square(image1 - image2)))

def IOU(image1, image2):
    src1 = (image1 >= 125)
    src2 = (image2 >= 125)

    overlap = src1 * src2
    union   = src1 + src2

    IOU = overlap.sum() / float(union.sum())
    return IOU

class TestToneProcessing(unittest.TestCase):
    """
    영상의 Color Tone을 변경하여, 분위기를 바꾸는 연산
    영상 내 연산은 Machine의 Spec에 따라 연산 오류가 다르게 날 수 있으므로,
    근사값(MSE, Mean-Squared-Error)을 통해 통과 여부를 결정

    1) colorTone     --  None-Linear Function(exp)로 ColorTone 값을 변경
        * test_colorTone_gamma_red_up     : gamma = (1.2,0.8,0.9)로 연산
        * test_colorTone_gamma_blue_up    : gamma = (0.8,1.0,1.2)로 연산

    2) monoTone    --  Color에서 Gray(mono)로 변경
        * test_monoTone     : monoTone 적용

    3) customTone  --  주어진 Look-Up-Table을 이용하여 변경
        * test_customTone_sepia   : sepia.lut을 적용

    4) saturationAdjust -- Saturation 값을 변경하여 색의 선명함 변경
        * test_saturationAdjust_with_g_06 : gamma = 0.6 을 적용
        * test_saturationAdjust_with_g_15 : gamma = 1.5 을 적용

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

    def test_colorTone_gamma_red_up(self):
        result = colorTone(self.image, 1.2,0.8,0.9)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(tone_dir,"test_colorTone_gamma_red_up.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertGreaterEqual(mse, 1.)

    def test_colorTone_gamma_blue_up(self):
        result = colorTone(self.image, 0.8,1.0,1.2)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(tone_dir,"test_colorTone_gamma_blue_up.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertGreaterEqual(mse, 1.)

    def test_monoTone(self):
        result = monoTone(self.image)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(tone_dir,"test_monoTone.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertGreaterEqual(mse, 1.)

    def test_customTone_sepia(self):
        lut_path = os.path.join(lut_dir,"sepia.lut")
        lut = read_lut(lut_path)

        result = customTone(self.image,lut)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(tone_dir,"test_customTone_sepia.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertGreaterEqual(mse, 1.)

    def test_saturationAdjust_with_g_06(self):
        result = saturationAdjust(self.image,0.6)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(tone_dir,"test_saturationAdjust_with_g_06.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertGreaterEqual(mse, 1.)

    def test_saturationAdjust_with_g_15(self):
        result = saturationAdjust(self.image,1.5)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")
        # load correct answer
        testpath = os.path.join(tone_dir,"test_saturationAdjust_with_g_15.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertGreaterEqual(mse, 1.)

if __name__ == "__main__":
    unittest.main(verbosity=2)
