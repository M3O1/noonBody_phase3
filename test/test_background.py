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
texture_dir = os.path.join(input_dir, "texture/")

background_dir = os.path.join(filter_dir, "background/")

def MSE(image1, image2):
    return np.sqrt(np.mean(np.square(image1 - image2)))

def IOU(image1, image2):
    src1 = (image1 >= 125)
    src2 = (image2 >= 125)

    overlap = src1 * src2
    union   = src1 + src2

    IOU = overlap.sum() / float(union.sum())
    return IOU

class TestBackGroundProcessing(unittest.TestCase):
    """
    영상에 질감, 조명 효과등을 주어서 보다 몸의 Shape에 집중할 수 있도록 도와줌
    영상 내 연산은 Machine의 Spec에 따라 연산 오류가 다르게 날 수 있으므로,
    근사값(MSE, Mean-Squared-Error)을 통해 통과 여부를 결정

    1) mergeBackgrounds   -- profile 외 영역을 지정한 Background로 붙이는 연산
        * test_mergeBackgrounds_with_cemento : cemento 이미지를 배경으로 붙이기
        * test_mergeBackgrounds_with_canvas  : canvas  이미지를 배경으로 붙이기

    2) vignetteEffect     --  이미지에 조명효과를 부과하는 연산
        * test_vignetteEffect_with_center_emphasized : args = (0.8,1.2,0.8,1.2)로 연산
        * test_vignetteEffect_with_side_emphasized   : args = (1.2,0.8,1.2,0.8)로 연산

    3) textureEffect      --  이미지에 질감효과를 부과하는 연산
        * test_textureEffect_with_cemento_str_05     : cemento 질감에 strength=0.5로 연산
        * test_textureEffect_with_canvas_str_05      : canvas 질감에 strength=0.5로 연산
        * test_textureEffect_with_cemento_str_12     : cemento 질감에 strength=1.2로 연산
        * test_textureEffect_with_canvas_str_12      : canvas 질감에 strength=1.2로 연산

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

    def test_mergeBackgrounds_with_cemento(self):
        backround = read_image(os.path.join(texture_dir,'cemento.jpg'))
        mask = reverse_mask(self.profile)
        result = merge_background(self.image, mask, backround)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"test_mergeBackgrounds_with_cemento.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_mergeBackgrounds_with_canvas(self):
        backround = read_image(os.path.join(texture_dir,'canvas.jpg'))
        mask = reverse_mask(self.profile)
        result = merge_background(self.image, mask, backround)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"test_mergeBackgrounds_with_canvas.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_vignetteEffect_with_center_emphasized(self):
        result = vignetteEffect(self.image,0.8,1.2,0.8,1.2)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"vignetteEffect_with_center_emphasized.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_vignetteEffect_with_side_emphasized(self):
        result = vignetteEffect(self.image,1.2,0.8,1.2,0.8)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"vignetteEffect_with_side_emphasized.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_textureEffect_with_cemento_str_05(self):
        texture = read_profile(os.path.join(texture_dir,"cemento.jpg"))
        result = textureEffect(self.image,texture,strength=0.5)

        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"textureEffect_with_cemento_str_05.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_textureEffect_with_canvas_str_05(self):
        texture = read_profile(os.path.join(texture_dir,"canvas.jpg"))
        result = textureEffect(self.image,texture,strength=0.5)

        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"textureEffect_with_canvas_str_05.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_textureEffect_with_cemento_str_12(self):
        texture = read_profile(os.path.join(texture_dir,"cemento.jpg"))
        result = textureEffect(self.image,texture,strength=1.2)

        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"textureEffect_with_cemento_str_12.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

    def test_textureEffect_with_canvas_str_12(self):
        texture = read_profile(os.path.join(texture_dir,"canvas.jpg"))
        result = textureEffect(self.image,texture,strength=1.2)

        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(background_dir,"textureEffect_with_canvas_str_12.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)
        self.assertLessEqual(mse, 10)

if __name__ == "__main__":
    unittest.main(verbosity=2)
