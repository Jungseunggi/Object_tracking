# Object_tracking

## 1. 논문 방법 및 내용 요약
<img src="https://user-images.githubusercontent.com/102225200/199154806-97f6915b-842f-40aa-830a-bdf369599cdf.png" width="300">


- 논문에 따르면 순서는 위 그림과 같음

  - 조건으로는 고정된 카메라여야 함 
  
### 1.1 Object detection

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


```
import cv2

cap = cv2.VideoCapture("highway.mp4")

# history를 통해 프레임간의 차이를 계산(따라서 카메라가 움직이면 다시 초기화),varThreshold를 통해 임계값 설정
object_detector = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50) 

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)

    <span style="color:red"># threshold를 통해 254이하는 모두 제거 따라서 회색은 삭제되고 흰색만 남음 -> 그림자 제거<span>
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    # 경계선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    for cnt in contours:
        # 경계선을 추출하면 배경에도 자잘한것들이 보여진다. 여기서 작은 면적들은 제거
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1,(0,0,255),1)


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
```

<img src="https://user-images.githubusercontent.com/102225200/199158258-755ae8cf-1784-45d1-8705-c11ac517ed9b.gif" width="400">

- 따라서 경계선, 그림자, 파라미터 등을 조절하여 정교하게 탐지 되도록 수정
 
- 하지만 여전히 배경에서 탐지되는 부분들이 존재

```
# 영상의 해상도는 1280*720으로 관심영역을 지정
roi = frame[340: 720,500: 800]
```
<img src="https://user-images.githubusercontent.com/102225200/199160412-372abee2-1bd4-43da-9b6a-5f84736b7f42.gif" width="400">

- 탐지할 부분을 설정하여 배경에서 탐지되 부분들을 제거

<img src="https://user-images.githubusercontent.com/102225200/199161329-9c2b61fd-1937-4120-b915-5cbfe39ce27c.gif" width="400">

- 이렇게 마스크를 씌어진 영상에서 객체를 박스처리

## 3. Object tracking을 접하게된 계기

Computer vision에 대해 관심이 많았는데 지식이 많이 부족하다고 느낌. computer vision하면 떠오르는게 object detection과 tracking이 가장 대표적이라고 생각함. 이를 통해 조금 더 성장하고 싶었음.

## 4. 후기

이번에 참고한 논문은 object tracking의 기본이라고 알고 있어서 참고 함. 하지만 이해는 되어도 어떻게 구현해야될지 어려운 부분들이 많음. 
따라서 유투브를 통해 직접 해보면서 어떻게 작동되는지 이해하는 방향으로 진행했음. 이 후 더 보완하여 객체탐지 모델과 연동을 시켜서 car1, car2, motorcycle1, motorcycle2 ... 이런식으로 카테고리도 분류하면서 추적하는 방향으로 진행하고 싶음.



