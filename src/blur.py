import numpy as np
import cv2

def avgBlur(image, ksize=11):
    """ apply average blur in image

    (ksize,ksize) 내의 픽셀 값 평균을 중앙 pixel값으로 대체하는 연산
    가장 기초적인 형태의 Blur지만, 구리기 그지없음

    Keyword arguments:
    ksize -- the blur size, 값이 클수록 blur 효과가 더 커짐
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]
    blurred = cv2.blur(rgb, (ksize,ksize))
    result = np.concatenate([blurred,alpha],axis=-1)
    return result

def normBlur(image, ksize=11, sigmaX=0):
    """ apply normalization blur in image

    (ksize,ksize) 내의 픽셀 값을 Gaussian 분포에 따라 곱해준 평균을
    중앙 pixel값으로 대체하는 연산
    avgBlur보다 훨씬 Smooth하게 변화하는 Blur

    Keyword arguments:
    ksize -- the blur size, 값이 클수록 blur 효과가 더 커짐
    sigmaX -- Normalization 분포의 sigmaX 값.
    """
    if ksize % 2 ==0:
        ksize += 1 # ksize는 홀수여야 함

    rgb, alpha = image[:,:,:3], image[:,:,3:]
    blurred = cv2.GaussianBlur(rgb, (ksize,ksize), sigmaX)
    result = np.concatenate([blurred, alpha],axis=-1)
    return result

def medianBlur(image, ksize=11):
    """ apply median blur in image

    (ksize,ksize) 내의 픽셀 값의 중앙값을 중앙 pixel값으로 대체하는 연산
    위 두 Blur와 다른 느낌의 Blur를 생산함. 이런거 좋아하는 사람 있을듯

    Keyword arguments:
    ksize -- the blur size, 값이 클수록 blur 효과가 더 커짐
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]
    blurred = cv2.medianBlur(rgb, ksize)
    result = np.concatenate([blurred, alpha],axis=-1)
    return result

def bltBlur(image, d=11, sigmaColor=75, sigmaSpace=75):
    """ apply bilateral blur in image

    normBlur는 중심 화소에서의 거리에 따른 가중치를 적용한 MASK를
    사용해서 영상을 부드럽게 만들어줌
    bltBlur도 normBlur처럼 가중치를 적용한 MASK를 사용하지만
    결정적으로 다른 점은 가중치에 중심 화소에서의 거리뿐만 아니라
    중심 화소와의 밝기 차이도 고려한다는 점이 차이.
    좀 더 노이즈에 강하지만

    Keyword arguments:
    d -- 필터링에 이용하는 이웃한 픽셀의 지름을 정의 불가능한경우 sigmaspace 를 사용
    sigmaColor -- 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다
    sigmaSpace -- 시그마 필터를 조정. 값이 클수록 긴밀하게 주변 픽셀에 영향을 미침.
                  d>0 이면 영향을 받지 않고, 그 외에는 d 값에 비례한다
    """
    rgb, alpha = image[:,:,:3], image[:,:,3:]
    blurred = cv2.bilateralFilter(rgb, d, sigmaColor, sigmaSpace)
    result = np.concatenate([blurred, alpha],axis=-1)
    return result
