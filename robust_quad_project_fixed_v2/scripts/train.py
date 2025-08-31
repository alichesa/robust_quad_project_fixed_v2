import os, argparse, yaml
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from quad_data.quad_dataset import QuadKeypointDataset
from models.unet import UNet
from losses.losses import mse_loss, bce_loss, homography_consistency_loss


def fgsm_attack(x, loss, epsilon=0.015):
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = (x + epsilon * grad.sign()).clamp(0, 1).detach()
    return x_adv


def train_one_epoch(model, loader, optim, device, cfg, epoch=0, log_interval=20):
    model.train()
    total = 0.0
    for step, batch in enumerate(loader, 1):
        # 打印当前batch的信息，确认数据加载是否正常
        print(f"Batch {step}/{len(loader)}: ", batch['image'].shape)

        x = batch["image"].to(device).requires_grad_(True)
        y_heat = batch["heatmap"].to(device)
        y_vis = batch["visibility"].to(device)

        heat, vis = model(x)
        Lh = mse_loss(heat, y_heat)
        Lv = bce_loss(vis, y_vis)

        # 计算 homography 一致性损失
        B, K, H, W = y_heat.shape
        with torch.no_grad():
            xs = torch.arange(W, device=device).float()
            ys = torch.arange(H, device=device).float()
            xv, yv = torch.meshgrid(ys, xs, indexing="ij")
            y_heat_norm = torch.softmax(y_heat.view(B, K, -1), dim=-1)
            grid = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=-1).float()  # HWx2 (y,x)
            coords = torch.einsum('bkc,cd->bkd', y_heat_norm, grid)  # y,x
            coords = coords[..., [1, 0]]  # -> x,y
        Lhomo = homography_consistency_loss(heat, coords, weight=1.0)

        loss = Lh + float(cfg["loss"]["w_visibility"]) * Lv + float(cfg["loss"]["w_homography"]) * Lhomo

        # 可选 FGSM 对抗
        if cfg["train"]["adv"]["use_fgsm"]:
            adv_x = (x + float(cfg["train"]["adv"]["epsilon"]) * torch.autograd.grad(loss, x, retain_graph=True)[
                0].sign()).clamp(0, 1).detach()
            heat2, vis2 = model(adv_x)
            Lh2 = mse_loss(heat2, y_heat)
            Lv2 = bce_loss(vis2, y_vis)
            loss = 0.5 * loss + 0.5 * (Lh2 + float(cfg['loss']['w_visibility']) * Lv2)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item()

        # --- 关键：step 级别打印 ---
        if (step == 1) or (step % log_interval == 0) or (step == len(loader)):
            print(f"Epoch {epoch + 1:03d} | Step {step:04d}/{len(loader):04d} "
                  f"| loss={loss.item():.4f} (Lh={Lh.item():.4f}, Lv={Lv.item():.4f}, Lhomo={Lhomo.item():.4f})")

    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        x = batch["image"].to(device)
        y_heat = batch["heatmap"].to(device)
        heat, _ = model(x)
        loss = F.mse_loss(heat, y_heat)
        total += loss.item()
    return total / len(loader)


def main(args):
    cfg = yaml.safe_load(open(args.config, 'r'))

    # 确保输出目录存在
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"Output directory: {cfg['out_dir']}")

    # 数据集加载
    ds_train = QuadKeypointDataset(cfg["data"]["images_dir"], cfg["data"]["ann_file"],
                                   img_size=tuple(cfg["data"]["img_size"]), sigma=cfg["data"]["heatmap_sigma"],
                                   train=True, split=cfg["data"]["train_val_split"])
    ds_val = QuadKeypointDataset(cfg["data"]["images_dir"], cfg["data"]["ann_file"],
                                 img_size=tuple(cfg["data"]["img_size"]), sigma=cfg["data"]["heatmap_sigma"],
                                 train=False, split=cfg["data"]["train_val_split"])

    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=cfg["train"]["workers"])
    dl_val = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=cfg["model"]["in_channels"], base=cfg["model"]["base_channels"],
                 num_keypoints=cfg["model"]["num_keypoints"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                              weight_decay=float(cfg["train"]["weight_decay"]))

    best = float("inf")
    log_interval = int(cfg["train"].get("log_interval", 20))

    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']} training...")

        # 训练当前epoch
        tr = train_one_epoch(model, dl_train, optim, device, cfg, epoch=epoch, log_interval=log_interval)

        # 验证集损失
        val = validate(model, dl_val, device)
        print(f"[{epoch + 1:03d}/{cfg['train']['epochs']}] train={tr:.4f}  val={val:.4f}")

        # 保存最佳模型
        if val < best:
            best = val
            best_save_path = os.path.join(cfg["out_dir"], "best.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"Saved best model at: {best_save_path}")

        # 每个epoch保存权重
        epoch_save_path = os.path.join(cfg["out_dir"], f"epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Saved model at: {epoch_save_path}")

        # 保存最后一次模型
        last_save_path = os.path.join(cfg["out_dir"], "last.pth")
        torch.save(model.state_dict(), last_save_path)
        print(f"Saved last model at: {last_save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    main(args)
