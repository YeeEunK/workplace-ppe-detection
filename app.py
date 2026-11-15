from io import BytesIO
import hashlib
import cv2
from pathlib import Path
from tempfile import TemporaryDirectory

from video_analysis import process_video

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from safety_logic import associate, overall

st.set_page_config(page_title='작업장 안전 탐지', layout='wide')
st.title('작업장 보호구 및 위험구역 탐지')
st.caption('교육용 프로토타입 · 결과는 참고용이며 실제 안전 상태를 보장하지 않습니다.')

@st.cache_resource
def load_model():
    return YOLO(str(Path(__file__).parent / 'models' / 'best.pt'))

with st.sidebar:
    st.header('탐지 설정')
    confidence = st.slider('탐지 기준값', 0.05, 0.90, 0.25, 0.05)
    zone_enabled = st.checkbox('위험구역 사용', value=False)
    if zone_enabled:
        st.caption('이미지에서 차지하는 비율(%)로 직사각형 영역을 지정합니다.')
        left = st.slider('왼쪽 경계', 0, 99, 20)
        right = st.slider('오른쪽 경계', 1, 100, 80)
        top = st.slider('위쪽 경계', 0, 99, 40)
        bottom = st.slider('아래쪽 경계', 1, 100, 90)

input_mode = st.radio('분석할 파일', ['이미지', '동영상'], horizontal=True)

if input_mode == '이미지':
    uploaded = st.file_uploader('작업장 이미지 업로드', type=['jpg', 'jpeg', 'png'])
    if uploaded:
        if uploaded.size > 15 * 1024 * 1024:
            st.error('15MB 이하의 이미지를 선택해 주세요.')
            st.stop()
        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(uploaded.getvalue()))).convert('RGB')
            if image.width * image.height > 25_000_000:
                st.error('이미지 크기가 너무 큽니다. 2,500만 픽셀 이하 이미지를 사용해 주세요.')
                st.stop()
        except (UnidentifiedImageError, OSError, ValueError):
            st.error('이미지를 열 수 없습니다. 다른 JPG 또는 PNG 파일을 선택해 주세요.')
            st.stop()
        if zone_enabled and (left >= right or top >= bottom):
            st.error('위험구역의 왼쪽은 오른쪽보다, 위쪽은 아래쪽보다 작아야 합니다.')
            st.stop()
        polygon = [(left*image.width/100, top*image.height/100), (right*image.width/100, top*image.height/100), (right*image.width/100, bottom*image.height/100), (left*image.width/100, bottom*image.height/100)] if zone_enabled else []
        preview = image.copy()
        if polygon:
            ImageDraw.Draw(preview).polygon(polygon, outline='red', width=max(3, image.width//400))
        st.image(preview, caption='입력 이미지와 위험구역', use_container_width=True)
        if st.button('안전 상태 분석', type='primary'):
            try:
                with st.spinner('이미지를 분석하고 있습니다...'):
                    result = load_model().predict(image, imgsz=960, conf=confidence, verbose=False)[0]
                detections = [{'name': result.names[int(b.cls.item())], 'confidence': float(b.conf.item()), 'box': b.xyxy[0].tolist()} for b in result.boxes]
                people, unassigned = associate(detections, polygon)
                status = overall(people, unassigned, zone_enabled)
                rendered = Image.fromarray(result.plot()[:, :, ::-1])
                draw = ImageDraw.Draw(rendered)
                if polygon:
                    draw.polygon(polygon, outline='red', width=max(3, image.width//400))
                for p in people:
                    x,y=p['foot']; radius=max(4,image.width//220)
                    draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill='red' if p['intrusion'] else 'cyan')
                st.subheader(f'전체 상태: {status}')
                a,b,c=st.columns(3)
                a.metric('탐지된 작업자', len(people))
                b.metric('PPE 미착용 탐지 작업자', sum(any(p[k]=='미착용' for k in ('helmet','vest')) for p in people))
                c.metric('위험구역 침입', sum(p['intrusion'] for p in people) if zone_enabled else '미설정')
                st.image(rendered, caption='탐지 결과 · 점은 작업자 박스의 하단 중심', use_container_width=True)
                if people:
                    st.subheader('작업자별 결과')
                    for i,p in enumerate(people,1):
                        st.write(f"작업자 {i}: 안전모 {p['helmet']} / 안전조끼 {p['vest']} / 위험구역 {'침입' if p['intrusion'] else ('밖' if zone_enabled else '미설정')}")
                else:
                    st.info('작업자가 탐지되지 않았습니다. 이미지와 탐지 기준값을 확인해 주세요.')
                if unassigned:
                    st.warning(f'작업자와 연결하지 못한 보호구 탐지 {len(unassigned)}개가 있습니다.')
                st.caption('보호구와 작업자 연결은 박스 위치에 근거한 추정입니다. 가림, 작은 객체, 겹침으로 오판할 수 있습니다.')
            except Exception as exc:
                st.error(f'분석을 완료하지 못했습니다: {exc}')

else:
    st.caption('10초 이하 MP4 · 20MB 이하 · 초당 최대 3장면을 분석합니다. 원본 소리는 결과에 포함되지 않습니다.')
    st.caption('고정된 카메라 영상에 맞춘 기능입니다. 위험구역은 영상 화면에서 고정됩니다.')
    video = st.file_uploader('작업장 동영상 업로드', type=['mp4'])
    if video:
        video_key = (hashlib.sha256(video.getvalue()).hexdigest(), confidence, zone_enabled, (left, right, top, bottom) if zone_enabled else None)
        if st.session_state.get('video_key') != video_key:
            st.session_state.pop('video_result', None)
            st.session_state.pop('video_events', None)
            st.session_state['video_key'] = video_key
        if video.size > 20 * 1024 * 1024:
            st.error('20MB 이하 동영상을 선택해 주세요.')
            st.stop()
        if zone_enabled and (left >= right or top >= bottom):
            st.error('위험구역의 왼쪽은 오른쪽보다, 위쪽은 아래쪽보다 작아야 합니다.')
            st.stop()
        st.video(video)
        if zone_enabled:
            with TemporaryDirectory(prefix='ppe-preview-') as tmp:
                source = Path(tmp) / 'preview.mp4'
                source.write_bytes(video.getvalue())
                capture = cv2.VideoCapture(str(source))
                try:
                    ok, frame = capture.read()
                finally:
                    capture.release()
                if ok:
                    height, width = frame.shape[:2]
                    scale = min(1.0, 1280 / max(width, height))
                    if scale < 1:
                        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
                    preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    w, h = preview.size
                    ImageDraw.Draw(preview).rectangle((left*w/100, top*h/100, right*w/100, bottom*h/100), outline='red', width=max(3, w//300))
                    st.image(preview, caption='위험구역 미리보기 · 영상의 첫 장면', use_container_width=True)
                else:
                    st.warning('영상의 첫 장면을 읽을 수 없어 위험구역을 미리 보여주지 못했습니다.')
        st.caption('장면을 간격을 두고 분석하므로 그 사이의 상황은 놓칠 수 있습니다.')
        if st.button('동영상 분석 시작', type='primary'):
            try:
                bounds = (left, right, top, bottom) if zone_enabled else None
                with TemporaryDirectory(prefix='ppe-video-') as tmp:
                    root = Path(tmp)
                    source = root / 'input.mp4'
                    source.write_bytes(video.getvalue())
                    bar = st.progress(0, text='동영상을 분석하고 있습니다. 잠시 기다려 주세요.')
                    events, last_seen = [], {}
                    def save_event(seconds, labels, frame):
                        fresh = [label for label in labels if seconds - last_seen.get(label, -10) >= 1.0]
                        if fresh:
                            for label in fresh:
                                last_seen[label] = seconds
                            h, w = frame.shape[:2]
                            scale = min(1.0, 960 / max(w, h))
                            small = cv2.resize(frame, (max(1, int(w*scale)), max(1, int(h*scale)))) if scale < 1 else frame
                            ok, image = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ok:
                                events.append((seconds, fresh, image.tobytes()))
                    count = process_video(source, root/'intermediate.mp4', root/'result.mp4', load_model(), confidence, bounds,
                                          progress=lambda x: bar.progress(x, text='동영상을 분석하고 있습니다.'), event_callback=save_event)
                    st.session_state['video_result'] = (root/'result.mp4').read_bytes()
                    st.session_state['video_events'] = events
                    bar.progress(1.0, text=f'분석 완료 · {count}개 장면')
            except Exception as exc:
                st.session_state.pop('video_result', None)
                st.session_state.pop('video_events', None)
                st.error(f'영상 분석을 완료하지 못했습니다: {exc}')
        if 'video_result' in st.session_state:
            events = st.session_state.get('video_events', [])
            labels = {label for _, labels, _ in events for label in labels}
            st.subheader('영상 분석 요약')
            st.write('안전모 미착용: ' + ('감지됨' if '안전모 미착용 감지' in labels else '감지되지 않음'))
            st.write('안전조끼 미착용: ' + ('감지됨' if '안전조끼 미착용 감지' in labels else '감지되지 않음'))
            st.write('위험구역 침입: ' + ('감지됨' if '위험구역 침입 감지' in labels else '감지되지 않음') if zone_enabled else '위험구역 침입: 미설정')
            st.caption('일부 장면을 추출해 분석하므로 감지되지 않았더라도 미착용이나 침입이 없었다는 뜻은 아닙니다. 확인할 장면은 같은 유형당 1초 간격으로 표시합니다.')
            if events:
                st.subheader('확인이 필요한 장면')
                for seconds, labels, image in events:
                    st.image(image, caption=f"{seconds:.1f}초 · {', '.join(labels)}", use_container_width=True)
            else:
                if zone_enabled:
                    st.info('분석한 장면에서 미착용이나 위험구역 침입이 감지되지 않았습니다.')
                else:
                    st.info('분석한 장면에서 보호구 미착용이 감지되지 않았습니다. 위험구역 침입은 설정하지 않아 확인하지 않았습니다.')
            st.video(st.session_state['video_result'])
            st.download_button('결과 동영상 다운로드', st.session_state['video_result'],
                               file_name='ppe-analysis.mp4', mime='video/mp4')
