import numpy as np
import cv2
from image_utils import merge_images

def merge_background(image, mask, background):
    """ 이미지에 배경 이미지를 합침

    Keyword arguments:
    background : 배경 이미지, width, Channel수는 image와 동일해야 함
    """
    height, width = image.shape[:2]
    bg_image = cv2.resize(background,(width, height))

    result = merge_images(bg_image, image, mask)
    return result

def vignetteEffect(image, Xside, Xcenter, Yside, Ycenter):
    """ 이미지에 조명효과를 부여

    Keyword arguments:
    Xside   : x축 방향의 양 Side에서의 조명 가중치
    Yside   : y축 방향의 양 Side에서의 조명 가중치
    Xcenter : x축 방향의 center에서의 조명 가중치
    Ycenter : y축 방향의 center에서의 조명 가중치
    """
    Xside = float(Xside); Xcenter = float(Xcenter)
    Yside = float(Yside); Ycenter = float(Ycenter)
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    height, width = rgb.shape[:2]
    vignette_mask = create_vignette_mask(height, width, Xside, Xcenter, Yside, Ycenter)
    applied = np.clip(rgb * vignette_mask, 0, 250).astype('uint8')

    result = np.concatenate([applied,alpha],axis=-1)
    return result

def create_vignette_mask(height, width, Xside, Xcenter, Yside, Ycenter):
    """ 조명효과의 가중치 mask을 return

    Keyword arguments:
    height  : mask의 height
    width   : mask의 width
    Xside   : x축 방향의 양 Side에서의 조명 가중치
    Yside   : y축 방향의 양 Side에서의 조명 가중치
    Xcenter : x축 방향의 center에서의 조명 가중치
    Ycenter : y축 방향의 center에서의 조명 가중치
    """

    def vignette_func(n, min_d, max_d):
        return np.vectorize(lambda x : -(max_d-min_d)/((n//2)**2)*x*(x-n+1)+min_d)

    f_x = vignette_func(height, Xside, Xcenter)
    f_y = vignette_func(width, Yside, Ycenter)

    vertical = np.vstack([f_y(np.arange(0,width))]*height) # 수직 방향으로 가중치 연산
    horizontal = np.vstack([f_x(np.arange(0,height ))]*width).T # 수평 방향으로 가중치 연산

    vignette_mask = (vertical*horizontal) # 두 방향으로의 가중치 곱의 Transpose
    return np.expand_dims(vignette_mask,axis=-1)

def textureEffect(image, texture, strength=0.5):
    """ 이미지에 질감 효과를 덧입힘

    Keyword arguments:
    texture  -- 적용할 texture 이미지
    strength -- texture 효과의 정도

    """
    if len(texture.shape) == 3:
        texture = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)

    height, width = image.shape[:2]
    texture = cv2.resize(texture,(width, height))
    rgb, alpha = image[:,:,:3], image[:,:,3:]

    blank = np.zeros_like(texture, dtype=np.float)
    normed = cv2.normalize(texture, blank,
                           0.,1.,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    weights = (1 - strength) + 2 * strength * normed
    applied = np.clip(rgb * np.expand_dims(weights,-1),0,250).astype('uint8')

    result = np.concatenate([applied,alpha],axis=-1)
    return result
