## Image-Processing for Body Image

### Objective

* NoonBody 이미지 속에서 신체 부분을 구분 짓는 것
* NoonBody 이미지, 즉 신체가 부각되는 필터를 만드는 것

### Requirements

* opencv >= 3.1.0
* numpy >= 1.14.2
* python == 3.6.3
* keras >= 2.1.5

### process

아래와 같은 순서대로 이미지가 처리됨

![](https://preview.ibb.co/dRJMmy/process.png)

크게 두 부분으로, 

1) 이미지에서 신체영역을 구분짓는 Segmentation 부분
     => Detection 코드를 짜고 있는 중
2) 이미지를 보정하는 filter 부분

으로 나눌 수 있다.





