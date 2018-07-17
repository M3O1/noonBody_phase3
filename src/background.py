import numpy as np
import cv2

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
