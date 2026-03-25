import cv2
import numpy as np
import os

def ensure_video(video_path):
    """Проверка существования видео"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    cap.release()
    return True

def get_video_info(video_path):
    """Информация о видео"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    return fps, width, height, total_frames, duration