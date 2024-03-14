import os
import cv2
import numpy as np
import sys

def adjust_brightness(input_path, output_path, brightness_factor):
    # 이미지 읽기
    original_image = cv2.imread(input_path)
    print("processing:", input_path)
    # 밝기 조절
    adjusted_image = np.clip(original_image * brightness_factor, 0, 255).astype(np.uint8)

    # 조절된 이미지 저장
    cv2.imwrite(output_path, adjusted_image)

def process_images_in_directory(input_directory, output_directory, brightness_factor):
    # 입력 디렉토리 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(input_directory)

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 디렉토리 내의 각 이미지에 대해 밝기 조절 수행
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_directory, file_name)
            output_path = os.path.join(output_directory, file_name)
            adjust_brightness(input_path, output_path, brightness_factor)

if __name__ == "__main__":
    # 명령행 인수로 입력 폴더 및 출력 폴더 경로 받아오기
    if len(sys.argv) != 4:
        print("Usage: python frame_intensity.py input_folder output_folder brightness_factor")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    brightness_factor = float(sys.argv[3])

    # 이미지 처리 함수 호출
    process_images_in_directory(input_folder, output_folder, brightness_factor)
