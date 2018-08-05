import glob
import os
import inspect

import random
import math
import numpy as np
import cv2

from image_utils import *
from blur import *
from tone import *
from contrast import *
from background import *

import matplotlib.pyplot as plt

texture_dir = "../images/texture/"

'''
############################
# IMAGE-UTILS
    * filter_images
    * draw_images
    * apply_mask
    * split_params
    * get_filelist
    * apply_filterlist
############################
'''
def filter_images(imagepaths, funcs):
    profilepaths = [os.path.split(filepath)[0][:-5] + "profile/" + os.path.split(filepath)[1] for filepath in imagepaths]
    filtered = []

    for imagepath, profilepath in zip(imagepaths, profilepaths):
        image = read_image(imagepath)
        profile = read_profile(profilepath)
        image = apply_filterlist(image, profile, funcs)
        filtered.append(image)

    return filtered

def draw_images(images,refs=None,figsize=(10,10)):
    cols = int(math.sqrt(len(images)))
    rows = len(images)//cols + 1
    if refs:
        images = [np.concatenate((img1, img2), axis=1) for img1,img2 in zip(images,refs)]

    fig = plt.figure(figsize=figsize)
    for idx, image in enumerate(images):
        ax = fig.add_subplot(rows,cols,idx+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.imshow(image)
    plt.subplots_adjust(wspace=0, hspace=0)

def apply_mask(profile, mask_width=0,mask_reversed=False, mask_not=True):
    '''
    profile을 통해, filter의 적용 범위인 mask을 계산
    '''
    # mask를 적용안할 경우 뺌
    if mask_not:
        return np.ones_like(profile,dtype=np.uint8)*255

    # mask width에 따라 mask의 넓이 연산
    if mask_width>0:
        mask = expand_mask(profile,width=abs(mask_width))
    elif mask_width<0:
        mask = reduce_mask(profile,width=abs(mask_width))
    else:
        mask = profile

    # mask_reversed에 따라 mask의 뒤집음 여부
    if mask_reversed:
        mask = reverse_mask(mask)

    return mask

def split_params(params):
    '''
    paramter를 메소드 별로 쪼갬
    '''
    mask_params = { param:value for (param,value) in params.items() if 'mask_' in param}
    filter_params = { param:value for (param,value) in params.items() if not 'mask_' in param}
    return params, mask_params, filter_params

def get_imagepaths(input_dir):
    return glob.glob(os.path.join(input_dir,"image/*"))

def apply_filterlist(image, profile, funcs):
    for func in funcs:
        image = func[0](image,profile)
    return image

'''
############################
# BLUR
    * apply_avgBlur
    * apply_normBlur
    * apply_medianBlur
    * apply_bltBlur
############################
'''
def apply_avgBlur(ksize=11, mask_width=0, mask_reversed=False, mask_not=True):
    '''
    ksize : blur size, 값이 클수록 blur 효과가 더 커짐
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_normBlur(ksize=11, sigmaX=0, mask_width=0, mask_reversed=False, mask_not=True):
    '''
    ksize : blur size, 값이 클수록 blur 효과가 더 커짐
    sigmaX : Normalization 분포의 sigmaX 값(어떤 효과를 주는지 잘 모르겠음 보통 0으로 둠)
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_medianBlur(ksize=11, mask_width=0, mask_reversed=False, mask_not=True):
    '''
    ksize : the blur size, 값이 클수록 blur 효과가 더 커짐
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_bltBlur(d=11, sigmaColor=75, sigmaSpace=75, mask_width=0, mask_reversed=False, mask_not=True):
    '''
    d -- 필터링에 이용하는 이웃한 픽셀의 지름. 정의 불가능한경우 sigmaspace 를사용
    sigmaColor -- 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다
    sigmaSpace -- 시그마 필터를 조정. 값이 클수록 긴밀하게 주변 픽셀에 영향을 미침. d>0 이면 영향을 받지 않고, 그 외에는 d 값에 비례한다
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

'''
############################
# CONTRAST
    * apply_claheContrast
    * apply_sharpening
    * apply_gammaContrast
    * apply_autoGammaContrast
############################
'''

def apply_claheContrast(clipLimit=2., tileGridSize=11, mask_width=0, mask_reversed=True, mask_not=True):
    '''
    tileGridSize : tile은 균일화 단위, tile의 범위가 클수록 큰 범위에서의 균일화가 일어남
    clipLimit : tile 연산시, 극단적인 값을 배제하는 기준. 이 값을 넘어가는 경우는 그 영역은 다른 영역에 균일하게 배분하여 적용
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_sharpening(sharpness=0.3, ksize=11, mask_width=0, mask_reversed=True, mask_not=True):
    '''
    sharpness : 값이 클수록 경계선이 부각됨
    ksize : blur 계산할 때 이용되는 값 (블러효과가 클수록 경계선 부각됨)
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_gammaContrast(gamma=0.8, mask_width=0, mask_reversed=True, mask_not=True):
    '''
    gamma : 1.0을 기준으로 클수록 밝아짐
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_autoGammaContrast(thr=80, mask_width=0, mask_reversed=True, mask_not=True):
    '''
    thr : 목표 밝기 보정 값(ref: 0~255). 작을수록 어두워지고, 커질수록 밝아짐
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, profile, **filter_params),image, mask)

    return func, params

'''
############################
# TONE
    * apply_colorTone
    * apply_monoTone
    * apply_saturationAdjust
############################
'''

def apply_colorTone(r_gamma=1.0,g_gamma=1.0,b_gamma=1.0, mask_width=0, mask_reversed=False, mask_not=True):
    '''
    r_gamma : Red에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    g_gamma : Green에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    b_gamma : Blue에 해당하는 arg, 1.0을 기준으로 클수록 강조됨.
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_monoTone(mask_width=0, mask_reversed=False, mask_not=True):
    '''
    이미지를 흑백(mono)로 변경
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_saturationAdjust(gamma=0.8, mask_width=0, mask_reversed=False, mask_not=True):
    '''
    gamma -- 채도값을 변경하는 함수, 1보다 작을수록 채도값이 작아짐
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

'''
############################
# BACKGROUND
    * apply_background
    * apply_vignetteEffect
    * apply_textureEffect
############################
'''

def apply_background(bg_filename="canvas.jpg", mask_width=0, mask_reversed=False, mask_not=False):
    '''
    bg_filename : 적용할 background 이미지. images/texture/에 있는 파일 이름
    '''
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    params['name'] = 'merge_background'

    def func(image,profile):
        height, width = image.shape[:2]
        background = read_image(os.path.join(texture_dir,bg_filename))
        background = cv2.resize(background, (width, height))

        mask = apply_mask(profile, **mask_params)
        return merge_images(background,image, mask)
    return func, params

def apply_vignetteEffect(Xside=1.0, Xcenter=1.2, Yside=1.0, Ycenter=1.2, mask_width=0, mask_reversed=False, mask_not=False):
    """
    Xside   : x축 방향의 양 Side에서의 조명 가중치
    Yside   : y축 방향의 양 Side에서의 조명 가중치
    Xcenter : x축 방향의 center에서의 조명 가중치
    Ycenter : y축 방향의 center에서의 조명 가중치
    """
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params

def apply_textureEffect(texture='canvas.jpg', strength=0.5, mask_width=0, mask_reversed=False, mask_not=False):
    """
    texture  : 적용할 texture 이미지의 파일이름
    strength : texture 효과의 정도
    """
    params, mask_params, filter_params = split_params(locals()) # parameter를 전체, Mask 용, Filter 용으로 쪼갬
    func_name = inspect.stack()[0][3].replace("apply_","") # 적용할 parameter의 이름을 가져옴
    params['name'] = func_name

    def func(image,profile):
        texture = read_image(os.path.join(texture_dir,filter_params['texture']))
        filter_params['texture'] = texture
        mask = apply_mask(profile, **mask_params)
        return merge_images(globals()[func_name](image, **filter_params),image, mask)

    return func, params
