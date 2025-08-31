import os, json, math, random, argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def rand_color():
    return [int(x) for x in np.random.randint(0,255,3)]

def sample_quad(w,h, min_scale=0.25, max_scale=0.8):
    cx, cy = np.random.uniform(0.3*w,0.7*w), np.random.uniform(0.3*h,0.7*h)
    sw, sh = np.random.uniform(min_scale*w,max_scale*w), np.random.uniform(min_scale*h,max_scale*h)
    rect = np.array([
        [cx-sw/2, cy-sh/2],
        [cx+sw/2, cy-sh/2],
        [cx+sw/2, cy+sh/2],
        [cx-sw/2, cy+sh/2],
    ], dtype=np.float32)
    jitter = np.random.uniform(-0.15,0.15,(4,2))*np.array([sw,sh])
    quad = rect + jitter
    return quad

def is_convex(quad):
    s = 0
    for i in range(4):
        a=quad[i]; b=quad[(i+1)%4]; c=quad[(i+2)%4]
        ab=b-a; bc=c-b
        cross = ab[0]*bc[1]-ab[1]*bc[0]
        s += np.sign(cross)
    return abs(s)==4

def add_background(img):
    noise = np.random.randint(0,30,img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img

def draw_quad(img, quad, thickness=2):
    q = quad.astype(np.int32)
    cv2.polylines(img, [q], True, (0,0,0), thickness+2)
    cv2.polylines(img, [q], True, (220,220,220), thickness)
    return img

def add_overexposure(img, w, h):
    if np.random.rand() < 0.7:
        cx, cy = np.random.randint(0,w), np.random.randint(0,h)
        rad = np.random.randint(int(0.05*min(w,h)), int(0.25*min(w,h)))
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mask, (cx,cy), rad, 255, -1)
        blur = cv2.GaussianBlur(mask, (0,0), sigmaX=rad/2)
        for c in range(3):
            img[:,:,c] = np.clip(img[:,:,c] + (blur*0.5).astype(img.dtype), 0, 255)
    return img

def add_glare(img, w, h):
    if np.random.rand() < 0.6:
        x1, y1 = np.random.randint(0,w), np.random.randint(0,h)
        x2, y2 = np.clip([x1+np.random.randint(-w//3,w//3), y1+np.random.randint(-h//3,h//3)], [0,0], [w-1,h-1])
        overlay = img.copy()
        cv2.line(overlay, (x1,y1), (int(x2),int(y2)), (255,255,255), thickness=np.random.randint(5,20))
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    return img

def add_occluders(img, w, h, quads, visible_flags):
    num = np.random.randint(0,4)
    masks = []
    for _ in range(num):
        pts = np.random.randint(0, min(w,h), (np.random.randint(3,8), 2)).astype(np.int32)
        cv2.fillPoly(img, [pts], (np.random.randint(0,255),)*3)
        masks.append(pts)
    def point_in_poly(p, poly):
        return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
    for qi, q in enumerate(quads):
        for ci, p in enumerate(q):
            for poly in masks:
                if point_in_poly(p, poly):
                    visible_flags[qi][ci] = 0
    return img, visible_flags

def generate(args):
    out = Path(args.out)
    (out/"images").mkdir(parents=True, exist_ok=True)
    ann = {"images": [], "annotations": []}
    img_id = 0
    for i in tqdm(range(args.num), ncols=80):
        w, h = args.width, args.height
        img = np.zeros((h,w,3), np.uint8) + np.random.randint(150,210)
        img = add_background(img)
        nq = np.random.randint(1, 4)
        quads = []
        for _ in range(nq):
            for _try in range(50):
                q = sample_quad(w,h)
                area = cv2.contourArea(q.astype(np.float32))
                if area < 0.02*w*h: 
                    continue
                if is_convex(q):
                    quads.append(q)
                    break
        visible = [[1,1,1,1] for _ in range(len(quads))]
        for q in quads:
            img = draw_quad(img, q, thickness=2)
        img = add_overexposure(img, w, h)
        img = add_glare(img, w, h)
        img, visible = add_occluders(img, w, h, quads, visible)
        if np.random.rand() < 0.5:
            k = np.random.choice([3,5])
            img = cv2.GaussianBlur(img, (k,k), 0)
        if np.random.rand() < 0.7:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(50, 95)]
            _, enc = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(enc, 1)

        fn = f"{img_id:06d}.jpg"
        cv2.imwrite(str(out/"images"/fn), img)
        ann["images"].append({"id": img_id, "file_name": fn, "width": w, "height": h})
        for qi, q in enumerate(quads):
            ann["annotations"].append({
                "image_id": img_id,
                "quad": q.tolist(),
                "visible": visible[qi],
                "category_id": 1
            })
        img_id += 1
    with open(out/"annotations.json","w") as f:
        json.dump(ann, f, indent=2)
    print(f"Saved to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    generate(args)
