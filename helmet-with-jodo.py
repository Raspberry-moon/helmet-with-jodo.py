import cv2
import torch
import numpy as np
import pygame
import time
import threading
import smbus2
import pathlib
from pathlib import Path

# pathlib 경로 설정
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# BH1750 조도 센서 초기화
bus = smbus2.SMBus(1)
BH1750_ADDRESS = 0x23
BH1750_CONTINUOUS_HIGH_RES_MODE = 0x10

# 조도 값 읽기 함수
def read_light(addr=BH1750_ADDRESS):
    try:
        data = bus.read_i2c_block_data(addr, BH1750_CONTINUOUS_HIGH_RES_MODE, 2)
        light_level = ((data[1] + (256 * data[0])) / 1.2)
        return light_level
    except OSError as e:
        print(f"Error: {e}")
        return 1000  # 에러 발생 시 높은 값 반환

# 특정 조도 값 (임계값)
LIGHT_THRESHOLD = 10

# YOLOv5 모델 로드 (로컬 경로 사용)
model = torch.hub.load('.', 'custom', path='./best.pt', source='local', device='cuda' if torch.cuda.is_available() else 'cpu')

# Pygame 초기화 및 스피커 설정
pygame.mixer.init()
pygame.mixer.music.load("./helmet.mp3")

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 프레임 처리 주기 (30fps 기준)
frame_duration = 1.0 / 120

# 조도 센서 주기
light_check_interval = 0.5  # 0.5초에 한 번 조도 센서 확인
light_level = 100  # 초기 조도 값 (임의의 큰 값)
detecting = False  # 헬멧 감지 활성화 여부
window_open = False  # 창이 열려 있는지 여부

# 프레임 캡처 스레드
frame = None
ret = False

def capture_frames():
    global frame, ret
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

def check_light_level():
    global light_level, detecting
    while True:
        light_level = read_light()
        print(f"Current light level: {light_level} lx")
        if light_level < LIGHT_THRESHOLD and not detecting:
            detecting = True
            print("Rider detected, starting helmet detection...")
        elif light_level >= LIGHT_THRESHOLD and detecting:
            detecting = False
            print("Rider left, stopping helmet detection...")
        time.sleep(light_check_interval)

# 조도 센서 확인을 위한 스레드 시작
light_thread = threading.Thread(target=check_light_level)
light_thread.daemon = True
light_thread.start()

while True:
    start_time = time.time()
    
    if detecting:
        if frame is not None:
            # 이미지 크기 조정
            resized_frame = cv2.resize(frame, (320, 240))

            # YOLOv5로 이미지 처리
            results = model(resized_frame)

            # 결과에서 라벨 추출
            df_results = results.pandas().xyxy[0]
            labels = df_results['name'].to_numpy()
            confidences = df_results['confidence'].to_numpy()
            print(f"Detected labels: {labels}")
            print(f"Confidences: {confidences}")

            # 헬멧 미착용 감지 (정확도 0.8 이하)
            helmet_detected = False
            for label, confidence in zip(labels, confidences):
                if label == 'helmet' and confidence > 0.8:
                    helmet_detected = True
                    break

            if not helmet_detected:
                print("Helmet not detected or confidence too low!")
                if not pygame.mixer.music.get_busy():
                    print("Playing warning sound...")
                    pygame.mixer.music.play()
            else:
                if pygame.mixer.music.get_busy():
                    print("Stopping warning sound...")
                    pygame.mixer.music.stop()

            # 결과 이미지 렌더링
            annotated_frame = np.squeeze(results.render())
            annotated_frame = cv2.resize(annotated_frame, (640, 480))  # 원래 크기로 복원
            
            cv2.imshow('Helmet Detection', annotated_frame)
            window_open = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        if window_open:
            cv2.destroyWindow('Helmet Detection')
            window_open = False

    # 프레임 처리 시간 계산
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_duration - elapsed_time)
    time.sleep(0.02)
    
cap.release()
cv2.destroyAllWindows()
