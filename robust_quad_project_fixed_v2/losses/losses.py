import torch, cv2, numpy as np
import torch.nn.functional as F

def mse_loss(pred, target): return F.mse_loss(pred, target)
def bce_loss(pred, target): return F.binary_cross_entropy(pred, target)

def softargmax_2d(heat):
    B,K,H,W = heat.shape
    heat = heat.view(B,K,-1)
    heat = F.softmax(heat, dim=-1)
    xs = torch.linspace(0, W-1, W, device=heat.device)
    ys = torch.linspace(0, H-1, H, device=heat.device)
    xv, yv = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=-1) # HWx2 (y,x)
    coords = torch.einsum('bkc,cd->bkd', heat, grid.float())     # y,x
    coords = coords[..., [1,0]]
    return coords

def homography_consistency_loss(pred_heat, target_corners, weight=1.0):
    B,K,H,W = pred_heat.shape
    coords = softargmax_2d(pred_heat)
    loss_value = 0.0
    for b in range(B):
        p = coords[b].detach().cpu().numpy().astype(np.float32)
        t = target_corners[b].detach().cpu().numpy().astype(np.float32)
        if (np.isnan(p).any() or np.isnan(t).any()):
            continue
        try:
            Hmat = cv2.getPerspectiveTransform(p, t)  # p->t
            grid = []
            for yy in np.linspace(0,1,5):
                for xx in np.linspace(0,1,5):
                    grid.append([xx,yy])
            grid = np.array(grid, np.float32)
            dst = cv2.perspectiveTransform(grid[None,...], Hmat)[0]
            d = np.mean(np.linalg.norm(dst - grid, axis=1))
            loss_value += d
        except Exception:
            continue
    if B>0:
        loss_value = loss_value / B
    return weight * torch.tensor(loss_value, device=pred_heat.device, dtype=pred_heat.dtype)
