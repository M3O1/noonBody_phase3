import numpy as np
import cv2

def colorTone(image, r_gamma=1.0, g_gamma=1.0, b_gamma=1.0):
    """ 이미지의 Color Tone을 변경

    Keyword arguments:
    r_gamma -- Red에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    g_gamma -- Green에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    b_gamma -- Blue에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    r_channel = gammaCorrection(rgb[:,:,0], r_gamma)
    g_channel = gammaCorrection(rgb[:,:,1], g_gamma)
    b_channel = gammaCorrection(rgb[:,:,2], b_gamma)
    applied =  np.stack([r_channel,g_channel,b_channel],axis=-1)

    result = np.concatenate([applied, alpha],axis=-1)
    return result

def gammaCorrection(channel, gamma):
    """ 이미지의 channel 하나를 지수함수 형태에 따라 변경
    """
    gamma = float(gamma)
    inv_g = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_g) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(channel, lut)

def monoTone(image):
    """ 이미지를 흑백(mono)로 변경
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    applied = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    result = np.concatenate([applied,alpha],axis=-1)
    return result

def customTone(image, lut):
    """ 이미지에 Custom된 Look-Up-Table을 적용

    Keyword arguments:
    lut -- Red에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    r_channel = cv2.LUT(rgb[:,:,0], lut[:,0])
    g_channel = cv2.LUT(rgb[:,:,1], lut[:,1])
    b_channel = cv2.LUT(rgb[:,:,2], lut[:,2])
    applied =  np.stack([r_channel,g_channel,b_channel],axis=-1)

    result = np.concatenate([applied, alpha],axis=-1)
    return result

def saturationAdjust(image, gamma=0.8):
    """ 이미지의 채도를 변경
    채도값은 특정한 색상의 가장 진한 상태를 100%로 하였을 때 진함의 정도.
    채도값이 0이면 무채색에 가까워짐

    Keyword arguments:
    gamma -- 채도값을 변경하는 함수, 1보다 작을수록 채도값이 작아짐
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = gammaCorrection(hsv[:,:,1], gamma)
    applied = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    result = np.concatenate([applied, alpha],axis=-1)
    return result
