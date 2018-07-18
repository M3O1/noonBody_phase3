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

mask_dir = os.path.join(filter_dir, "mask/")

def MSE(image1, image2):
    return np.sqrt(np.mean(np.square(image1 - image2)))

def IOU(image1, image2):
    src1 = (image1 >= 125)
    src2 = (image2 >= 125)

    overlap = src1 * src2
    union   = src1 + src2

    IOU = overlap.sum() / float(union.sum())
    return IOU

class TestMaskProcessing(unittest.TestCase):
    """
    Profile(사람의 영역)을 받아 Mask(Filter의 적용범위)를 산출하는 연산들을 테스트
    영상 내 연산은 Machine의 Spec에 따라 연산 오류가 다르게 날 수 있으므로,
    근사값(IOU, Intersection-Over-Union)을 통해 통과 여부를 결정

    1) reduce_mask     --  profile 영역을 width 만큼 줄이는 연산
        * test_reduce_mask_width_1 : 두께 1만큼 줄이기
        * test_reduce_mask_width_3 : 두께 3만큼 줄이기

    2) expand_mask     --  profile 영역을 width 만큼 늘이는 연산
        * test_expand_mask_width_1 : 두께 1만큼 늘이기
        * test_expand_mask_width_3 : 두께 3만큼 늘이기

    3) reverse_mask    -- profile 영역을 반전시키는 연산
        * test_reverse_mask        : profile 반전이 올바른지 확인

    4) extract_border  -- profile의 border 영역을 width만큼 추출하는 연산
        * test_extract_border_width_1 : 두께 1만큼 뽑아내기
        * test_extract_border_width_3 : 두께 3만큼 뽑아내기

    5) merge_images    -- image1과 image2를 mask기준으로 합치는 연산
        * test_merge_image_with_red_bg   : test이미지를 붉은색 배경과 합침

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

    def test_reduce_mask_width_1(self):
        result = reduce_mask(self.profile, 1)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_reduce_mask_width_1.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)

        self.assertGreaterEqual(iou, 0.95)

    def test_reduce_mask_width_3(self):
        result = reduce_mask(self.profile, 3)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_reduce_mask_width_3.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)
        self.assertGreaterEqual(iou, 0.95)

    def test_expand_mask_width_1(self):
        result = expand_mask(self.profile, 1)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_expand_mask_width_1.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)
        self.assertGreaterEqual(iou, 0.95)

    def test_expand_mask_width_3(self):
        result = expand_mask(self.profile, 3)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_expand_mask_width_3.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)
        self.assertGreaterEqual(iou, 0.95)

    def test_reverse_mask(self):
        result = reverse_mask(self.profile)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_reverse_mask.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)
        self.assertGreaterEqual(iou, 0.95)

    def test_extract_border_width_1(self):
        result = extract_border(self.profile,1)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_extract_border_width_1.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)
        self.assertGreaterEqual(iou, 0.95)

    def test_extract_border_width_3(self):
        result = extract_border(self.profile,3)
        self.assertTupleEqual(self.profile.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_extract_border_width_3.png")
        correct = read_profile(testpath)

        iou = IOU(correct, result)
        self.assertGreaterEqual(iou, 0.95)

    def test_merge_image_with_red_bg(self):
        bg = np.ones_like(self.image,dtype='uint8') * np.array([125,0,0,255],dtype='uint8')
        result = merge_images(self.image, bg, self.profile)
        self.assertTupleEqual(self.image.shape, result.shape,
                              "the shape of input and output should be same")

        # load correct answer
        testpath = os.path.join(mask_dir,"test_merge_images_with_red_bg.png")
        correct = read_image(testpath)

        mse = MSE(correct, result)

        self.assertLessEqual(mse, 10)

if __name__ == "__main__":
    unittest.main(verbosity=2)
