import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# 고정된 카메라에서 움직이는 물체를 감지
# object detection
# histroy 기록시간 카메라가 움직이면 다시 변경, varThreshold = 임계값 높이면 자잘한것들 제거되지만 탐지를 못할수 있음
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)  

while True:
    ret, frame = cap.read()
    
    # 관심영역 추출
    height, width, _ = frame.shape
    # 좌표의 기준(왼쪽 모서리기준(0,0) 앞에가 세로,뒤가 가로)
    roi = frame[340: 720,500: 800]

    # 여기서 마스크는 객체제외하고는 모두 검은색으로 변경, 객체를 흰색으로 변경
    mask = object_detector.apply(roi)

    

    # threshold를 통해 254이하는 모두 제거 따라서 회색은 삭제되고 흰색만 남음
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    # 경계선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 감지된 물체 리스트로 반환
    detections = []
    for cnt in contours:
        # 경계선을 추출하면 배경에도 자잘한것들이 보여진다. 여기서 작은 면적들은 제거
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1,(0,0,255),1)
            # 바운딩박스로 탐지
            # 하지만 이대로는 그림자까지 감지 되어버림
            x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),1)
            detections.append([x,y,w,h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        # X = (x+w)/2
        # Y = (y+h)/2
        # X,Y,id = box_id
        x, y, w, h, id = box_id
        # 박스 위에 텍스트 생성
        cv2.putText(roi,str(id),(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),1)
    

    cv2.imshow("Frame", frame)
    cv2.imshow("Massk", mask)
    cv2.imshow("roi", roi)
    # waitKey 시간이 짧을수록 프레임속도가 다른듯
    key = cv2.waitKey(10)
    # 종료버튼
    if key == ord('q'):
        break
    # 일시정지
    if key == ord('p'):
        cv2.waitKey(-1)
    # waitkey는 키입력 대기시간으로 0이면 무한대기 30이면 30초동안 입력이 없으면 종료
    # key = cv2.waitKey(30)
    # if key == 27:
    #     break

cap.release()
cv2.destroyAllWindows()
print(detections)