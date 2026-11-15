"""Validation-only analysis of no_vest predictions, with IoU 0.5 matching."""
from pathlib import Path
import argparse
import csv
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='안전조끼 미착용 검증 결과 분석')
parser.add_argument('--data', type=Path, default=ROOT / 'data/construction-ppe-5class',
                    help='YOLO 데이터셋 폴더 (images/val과 labels/val 포함)')
DATA = parser.parse_args().data
OUT = ROOT / 'results/vest_val_analysis'
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union else 0

model = YOLO(str(ROOT / 'models/best.pt'))
totals = {t: [0, 0, 0] for t in THRESHOLDS}
details = []
images = sorted((DATA / 'images/val').glob('*.jpg'))
assert len(images) == 100
for i, path in enumerate(images, 1):
    with Image.open(path) as im:
        width, height = im.size
    truth = []
    for line in (DATA / 'labels/val' / (path.stem + '.txt')).read_text().splitlines():
        cols = line.split()
        if int(cols[0]) == 4:
            x, y, w, h = map(float, cols[1:])
            truth.append(((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height))
    result = model.predict(str(path), conf=0.001, imgsz=960, device='cpu', verbose=False)[0]
    predictions = sorted([(float(b.conf[0]), tuple(map(float, b.xyxy[0].tolist()))) for b in result.boxes if int(b.cls[0]) == 4], reverse=True)
    for threshold in THRESHOLDS:
        used = set(); false_positives = []
        for score, box in predictions:
            if score < threshold: continue
            best, j = max(((iou(box, gt), k) for k, gt in enumerate(truth) if k not in used), default=(0, -1))
            if best >= 0.5: used.add(j)
            else: false_positives.append((score, box))
        misses = [box for j, box in enumerate(truth) if j not in used]
        totals[threshold][0] += len(used); totals[threshold][1] += len(false_positives); totals[threshold][2] += len(misses)
        if threshold == 0.25 and (misses or false_positives):
            details.append([path.name, len(truth), len(used), len(false_positives), len(misses)])
            with Image.open(path) as src:
                im = src.convert('RGB'); draw = ImageDraw.Draw(im)
                for j, box in enumerate(truth): draw.rectangle(box, outline='lime' if j in used else 'red', width=5)
                for score, box in false_positives:
                    draw.rectangle(box, outline='cyan', width=4)
                    draw.text((box[0], max(0, box[1]-16)), f'FP {score:.2f}', fill='cyan')
                im.thumbnail((1280,1280)); im.save(OUT / (path.stem + '.jpg'), quality=88)
    if i % 10 == 0: print(f'{i}/{len(images)} images', flush=True)
with (OUT / 'threshold_metrics.csv').open('w', newline='') as f:
    writer=csv.writer(f); writer.writerow(['threshold','TP','FP','FN','precision','recall'])
    for t, (tp,fp,fn) in totals.items():
        writer.writerow([t,tp,fp,fn,round(tp/(tp+fp),4) if tp+fp else 0,round(tp/(tp+fn),4) if tp+fn else 0])
        print(t,tp,fp,fn,flush=True)
with (OUT / 'image_errors.csv').open('w',newline='') as f:
    writer=csv.writer(f); writer.writerow(['image','ground_truth','TP','FP','FN']); writer.writerows(details)
