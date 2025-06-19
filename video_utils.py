import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)
    return cap

def save_video_writer(path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
