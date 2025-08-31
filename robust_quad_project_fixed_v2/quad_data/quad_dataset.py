import json, os, random
import cv2, numpy as np, torch
from torch.utils.data import Dataset
import albumentations as A


def gaussian_heatmap(H, W, x, y, sigma=3.0):
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    g = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma * sigma))
    return g


def clamp_quad_in_bounds(q, W, H):
    q = q.copy()
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    return q


class QuadKeypointDataset(Dataset):
    def __init__(self, images_dir, ann_file, img_size=(512, 512), sigma=3.0, train=True, split=0.9):
        self.images_dir = images_dir
        ann = json.load(open(ann_file, 'r'))
        by_img = {im["id"]: im for im in ann["images"]}
        items = {}
        for a in ann["annotations"]:
            items.setdefault(a["image_id"], []).append(a)
        self.samples = [{"image": by_img[k], "anns": v} for k, v in items.items()]
        n = len(self.samples);
        idx = list(range(n))
        random.Random(42).shuffle(idx);
        sp = int(n * split)
        if train:
            idx = idx[:sp]
        else:
            idx = idx[sp:]
        self.samples = [self.samples[i] for i in idx]
        self.H, self.W = img_size
        self.sigma = sigma

        # Build transforms with version-compat parameters
        try:
            pad = A.PadIfNeeded(self.H, self.W, border_mode=cv2.BORDER_CONSTANT, border_value=(200, 200, 200))
        except TypeError:
            pad = A.PadIfNeeded(self.H, self.W, border_mode=cv2.BORDER_CONSTANT, value=(200, 200, 200))
        try:
            comp = A.ImageCompression(quality_range=(55, 95), p=0.3)
        except TypeError:
            comp = A.ImageCompression(quality_lower=55, quality_upper=95, p=0.3)

        self.aug = A.Compose([
            A.LongestMaxSize(max_size=max(self.W, self.H)),
            pad,
            A.OneOf([A.MotionBlur(3), A.GaussianBlur(3), A.ISONoise()], p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(p=0.2),
            comp,
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, check_each_transform=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        path = os.path.join(self.images_dir, rec["image"]["file_name"])
        img = cv2.imread(path);
        if img is None: raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        quads = [clamp_quad_in_bounds(np.array(a["quad"], dtype=np.float32), W0, H0) for a in rec["anns"]]
        vis = [np.array(a["visible"], dtype=np.float32) for a in rec["anns"]]

        kps = []
        for q in quads:
            for p in q: kps.append((float(p[0]), float(p[1])))

        aug = self.aug(image=img, keypoints=kps)
        img = aug["image"];
        kps = aug["keypoints"]

        quads2, vis2 = [], []
        it = iter(kps)
        for q, v in zip(quads, vis):
            qq = np.array([next(it), next(it), next(it), next(it)], dtype=np.float32)
            quads2.append(qq);
            vis2.append(v)

        num_kp = 4;
        H, W = self.H, self.W
        heat = np.zeros((num_kp, H, W), np.float32)
        vis_map = np.zeros((num_kp,), np.float32)

        areas = [cv2.contourArea(q.astype(np.float32)) for q in quads2]
        if len(areas) == 0:
            center = (W // 2, H // 2)
            for c in range(num_kp):
                heat[c] = gaussian_heatmap(H, W, center[0], center[1], self.sigma)
            vis_map[:] = 0
        else:
            j = int(np.argmax(areas));
            q = quads2[j];
            v = vis2[j]
            for c in range(num_kp):
                x, y = q[c];
                heat[c] = gaussian_heatmap(H, W, x, y, self.sigma)
                vis_map[c] = v[c]

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        heat_t = torch.from_numpy(heat.copy()).float()
        vis_t = torch.from_numpy(vis_map.copy()).float()
        return {"image": img_t, "heatmap": heat_t, "visibility": vis_t}
