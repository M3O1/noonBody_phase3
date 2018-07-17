import numpy as np
import cv2

def read_image(image_path):
    """ Read image file in 32bit format(RGBA)
    Keyword arguments:
    image_path -- the path of image to read
    """
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return image

def read_profile(profile_path):
    """ Read profile file in  8bit format(GRAY)
    Keyword arguments:
    profile_path -- the path of profile to read
    """
    profile = cv2.imread(profile_path,0)
    return profile

def reduce_mask(profile, width=1):
    """ return : profile에서 width만큼 뺀 영역
    Keyword arguments:
    width -- border 주위로 줄어드는 크기
    """
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(profile, kernel, iterations=width)
    return mask

def expand_mask(profile, width=1):
    """ return : profile에서 width만큼 더한 영역
    Keyword arguments:
    width -- border 주위로 늘어나는 크기
    """
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(profile, kernel, iterations=width)
    return mask

def reverse_mask(mask):
    """ return the reversed value of image
    """
    return ~mask

def extract_border(profile, width=1):
    """ return : profile의 border 부분을 width * 2배만큼 추출한 영역
    Keyword arguments:
    width -- border 주위로 추출될 크기
    """
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(profile, kernel, iterations=width)
    erosion = cv2.erode(profile, kernel, iterations=width)
    borderZone = dilation - erosion
    return borderZone

def merge_images(src1, src2, mask):
    """ merge two images by mask

    Keyword arguments:
    src1 -- mask에 채울 이미지
    src2 -- mask 외 부분에 채울 이미지
    mask -- merge의 기준이 되는 영역
    """
    out1 = cv2.bitwise_and(src1, src1, mask=mask)
    out2 = cv2.bitwise_and(src2, src2, mask=~mask)
    result = cv2.add(out1, out2)
    return result
