# Object_tracking

## 1. 방법론
<img src="https://user-images.githubusercontent.com/102225200/199154806-97f6915b-842f-40aa-830a-bdf369599cdf.png" width="300">


- 논문에 따르면 순서는 위 그림과 같음

  - 조건으로는 고정된 카메라여야 함 
  
### 1.1 bject detection

<img src="https://user-images.githubusercontent.com/102225200/199154748-dcf2855c-b83b-4979-85a0-c89fdc176dbc.png" width="300">

<img src="https://user-images.githubusercontent.com/102225200/199146346-781420c5-ada7-4aa0-9f93-606961552418.png" width="300">

- 고정된 이미지는 배경으로 구분하여 물체의 이동에 따른 차이로 object detection을 하는 것


### 1.2 Object representation

<img src="https://user-images.githubusercontent.com/102225200/199155254-40b30d43-6e43-4bdd-a0af-1e9dfdc031ca.png" width="300">



### 1.3 Object tracking 

<img src="https://user-images.githubusercontent.com/102225200/199156641-1797dbc1-21d1-45e2-b1a5-4d12ccc0014d.png" width="300">


## 2. 구현

<img src="https://user-images.githubusercontent.com/102225200/199146117-087e63d4-d3d0-4833-85a3-2aa5c61c2210.gif" width="400">


```
import cv2

cap = cv2.VideoCapture("highway.mp4")

# 배경을 제거해줌으로 객체를 탐지
object_detector = cv2.createBackgroundSubtractorMOG2() 

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)

    cv2.imshow("Massk", mask)
```

<img src="https://user-images.githubusercontent.com/102225200/199146097-56d2f2b4-9a03-420e-9708-20ce14c78691.gif" width="400">


- 아직은 배경을 제대로 분리하지를 못함(그림자, 차선, 나무 등)

<img src="https://user-images.githubusercontent.com/102225200/199158258-755ae8cf-1784-45d1-8705-c11ac517ed9b.gif" width="400">

- 따라서 경계선, 그림자, 파라미터 등을 조절하여 정교하게 탐지 되도록 수정 

**참고논문 https://www.researchgate.net/publication/301775263_A_Survey_on_Moving_Object_Detection_and_Tracking_Techniques**

