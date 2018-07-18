import numpy as np
import cv2

def claheContrast(image, clipLimit=2., tileGridSize=11):
    """ apply CLAHE(constrast limited adaptive histogram equalization)

    이미지를 작은 tile 형태로 나누어 그 tile 안에서 Equalization을 적용하는 방식.
    하지만 tile은 작은 영역이다 보니 작은 노이즈(극단적으로 어둡거나, 밝은 영역)이
    있으면, 이것이 반영이 되어 원하는 결과를 얻을 수 없게 됨.
    이 문제를 피하기 위해서
    contrast Limit이라는 값을 적용하여 이 값을 넘어가는 경우는
    그 영역을 다른 영역에 균일하게 배분하여 적용

    Keyword arguments:
    clipLimit --
    tileGridSize --
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    clipLimit = int(clipLimit)
    tileGridSize = int(tileGridSize)
    clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=(tileGridSize, tileGridSize))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    applied = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    result = np.concatenate([applied, alpha],axis=-1)
    return result

def sharpening(image, sharpness=0.3, ksize=11):
    """ 이미지의 경계선을 부각시킴

    이미지를 Blurry하게 만든 것과 원래 이미지를 빼줌으로서,
    이미지의 경계 부분이 좀 더 부각이 되도록 함

    Keyword arguments:
    sharpness -- 클수록 경계선 부각
    ksize -- blur계산할 때 이용하는 값
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    blurred = cv2.GaussianBlur(rgb, (ksize,ksize), 0)
    applied = cv2.addWeighted(rgb, 1+sharpness, blurred, -sharpness,0)

    result = np.concatenate([applied, alpha],axis=-1)
    return result

def gammaContrast(image, gamma=0.8):
    """ 이미지의 brightness를 조정하여 대조를 강조

    Keyword arguments:
    gamma -- 1.0을 기준으로 클수록 밝아짐

    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    gamma = float(gamma)
    inv_g = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_g) * 255 for i in np.arange(0, 256)]).astype("uint8")
    applied = cv2.LUT(rgb, lut)

    result = np.concatenate([applied, alpha],axis=-1)
    return result

def autoGammaContrast(image, profile, thr=80):
    """ 영상 내 인물 부분의 평균 밝기에 따라 GammaContrast를 보정

    Keyword arguments:
    profile -- 영상 내 인물 부분
    thr     -- 목표 보정 수치. 작을수록 어두워지고, 커질수록 밝아짐
    """
    mean_value = get_brightness(image, profile)
    gamma = float(thr) / mean_value
    result = gammaContrast(image, gamma)
    return result

def get_brightness(image, mask):
    """ 이미지의 mask 영역 내 평균 밝기값을 return
    """
    rgb = image[:,:,:3]

    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    extracted = cv2.bitwise_and(gray, gray, mask=mask)

    mean_value = np.sum(extracted) / np.count_nonzero(mask)
    return mean_value
