"""Short, sampled video inference for the PPE demonstration."""
import math
import subprocess
import cv2
from safety_logic import associate, overall

MAX_SECONDS = 10
TARGET_FPS = 3


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('영상을 열 수 없습니다. 다른 MP4 파일을 선택해 주세요.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            raise ValueError('영상 길이나 프레임 정보를 읽을 수 없습니다.')
        if count / fps > MAX_SECONDS + 0.1:
            raise ValueError(f'{MAX_SECONDS}초 이하 영상을 선택해 주세요.')
        return fps, int(count)
    finally:
        cap.release()


def process_video(path, intermediate, output, model, confidence, bounds, progress=None):
    fps, count = inspect_video(path)
    stride = max(1, math.ceil(fps / TARGET_FPS))
    cap = cv2.VideoCapture(str(path))
    writer = None
    frame_index = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index / fps > MAX_SECONDS + 0.1:
                raise ValueError('영상이 10초를 초과합니다.')
            if frame_index % stride == 0:
                height, width = frame.shape[:2]
                if width * height > 1920 * 1080:
                    raise ValueError('영상 해상도가 너무 큽니다. 1920×1080 이하 영상을 사용해 주세요.')
                polygon = [(bounds[0]*width/100, bounds[2]*height/100),
                           (bounds[1]*width/100, bounds[2]*height/100),
                           (bounds[1]*width/100, bounds[3]*height/100),
                           (bounds[0]*width/100, bounds[3]*height/100)] if bounds else []
                result = model.predict(frame, imgsz=960, conf=confidence, verbose=False)[0]
                detections = [{'name':result.names[int(b.cls.item())], 'confidence':float(b.conf.item()),
                               'box':b.xyxy[0].tolist()} for b in result.boxes]
                people, unassigned = associate(detections, polygon)
                status = overall(people, unassigned, bool(bounds))
                rendered = result.plot()
                if polygon:
                    pts = [(int(x), int(y)) for x,y in polygon]
                    for i in range(4):
                        cv2.line(rendered, pts[i], pts[(i+1)%4], (0,0,255), max(2,width//400))
                cv2.putText(rendered, {'확인 필요':'CHECK'}.get(status, status), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if writer is None:
                    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'), fps / stride, (width,height))
                    if not writer.isOpened():
                        raise RuntimeError('결과 영상을 만들 수 없습니다.')
                writer.write(rendered)
                written += 1
            frame_index += 1
            if progress:
                progress(min(frame_index / count, 1.0))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not written:
        raise ValueError('분석할 장면을 읽지 못했습니다.')
    try:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(intermediate),
                        '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],
                       check=True, capture_output=True, timeout=90)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('결과 영상을 브라우저 재생용으로 변환하지 못했습니다.') from exc
    return written
