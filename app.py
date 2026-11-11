from io import BytesIO
from pathlib import Path

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
