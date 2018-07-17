import numpy as np
import cv2

def colorTone(image, r_gamma=1.0, g_gamma=1.0, b_gamma=1.0):
    """ 이미지의 Color Tone을 변경

    Keyword arguments:
    r_gamma -- Red에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    g_gamma -- Green에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    b_gamma -- Blue에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    """
    def channelTone(channel, gamma):
        """ 이미지의 channel 하나를 변경
        """
        gamma = float(gamma)
        inv_g = 1.0 / gamma
        lut = np.array([
            ((i / 255.0) ** inv_g) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(channel, lut)

    rgb, alpha = image[:,:,:3], image[:,:,3:]

    r_channel = channelTone(rgb[:,:,0], r_gamma)
    g_channel = channelTone(rgb[:,:,1], g_gamma)
    b_channel = channelTone(rgb[:,:,2], b_gamma)
    applied =  np.stack([r_channel,g_channel,b_channel],axis=-1)

    result = np.concatenate([applied, alpha],axis=-1)
    return result

def monoTone(image):
    """ 이미지를 흑백(mono)로 변경
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    applied = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    result = np.concatenate([applied,alpha],axis=-1)
    return result
