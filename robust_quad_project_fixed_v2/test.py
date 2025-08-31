import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse  # 导入 argparse 库
from torch.utils.data import DataLoader
from quad_data.quad_dataset import QuadKeypointDataset
from models.unet import UNet
from losses.losses import mse_loss, bce_loss, homography_consistency_loss


def fgsm_attack(x, loss, epsilon=0.015):
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = (x + epsilon * grad.sign()).clamp(0, 1).detach()
    return x_adv

def test_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y_heat = batch["heatmap"].to(device)
            y_vis = batch["visibility"].to(device)

            # 进行推理（预测）
            heat, vis = model(x)

            # 选择第一张图片的输出（批量大小为 1）
            heat = heat.squeeze(0).cpu().numpy()  # (K, H, W) 格式
            vis = vis.squeeze(0).cpu().numpy()  # 可见性图

            # 可视化：展示预测的热图
            plt.figure(figsize=(8, 8))
            for i in range(heat.shape[0]):
                plt.subplot(heat.shape[0], 1, i + 1)
                plt.imshow(heat[i], cmap="jet")
                plt.title(f"Predicted Heatmap {i + 1}")
                plt.colorbar()
            plt.show()

            # 如果需要，计算角点误差或其他评估指标

def visualize_results(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y_heat = batch["heatmap"].to(device)
            y_vis = batch["visibility"].to(device)

            # 预测结果
            heat, vis = model(x)

            # 可视化真实热图与预测热图
            plt.figure(figsize=(12, 6))

            # 真实热图
            plt.subplot(1, 2, 1)
            plt.imshow(y_heat.squeeze(0).cpu().numpy()[0], cmap="jet")  # 显示第一个角点的热图
            plt.title("Ground Truth Heatmap")
            plt.colorbar()

            # 预测热图
            plt.subplot(1, 2, 2)
            plt.imshow(heat.squeeze(0).cpu().numpy()[0], cmap="jet")  # 显示第一个角点的预测热图
            plt.title("Predicted Heatmap")
            plt.colorbar()

            plt.show()

def compute_keypoint_error(pred, gt):
    # 计算均方根误差（RMSE）
    pred_coords = np.array(pred)
    gt_coords = np.array(gt)
    return np.sqrt(np.mean((pred_coords - gt_coords) ** 2))

def main(args):
    # 加载配置
    cfg = yaml.safe_load(open(args.config,'r'))

    # 确保输出目录存在
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"Output directory: {cfg['out_dir']}")

    # 加载测试数据
    ds_test = QuadKeypointDataset(cfg["data"]["images_dir"],
                                  cfg["data"]["ann_file"],
                                  img_size=tuple(cfg["data"]["img_size"]),
                                  sigma=cfg["data"]["heatmap_sigma"],
                                  train=False, split=cfg["data"]["train_val_split"])

    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=cfg["train"]["workers"])

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=cfg["model"]["in_channels"],
                 base=cfg["model"]["base_channels"],
                 num_keypoints=cfg["model"]["num_keypoints"]).to(device)

    # 加载训练好的权重
    model.load_state_dict(torch.load(os.path.join(cfg["out_dir"], "best.pth")))
    model.eval()  # 设置为评估模式

    # 测试模型
    print("Testing the model...")
    test_model(model, dl_test, device)

    # 可视化结果
    print("Visualizing results...")
    visualize_results(model, dl_test, device)

    # 计算角点误差（如果需要）
    # 你可以添加代码来计算角点的 RMSE 或其他评估指标
    # pred_coords 和 gt_coords 是你从模型输出和真实数据中获得的角点
    # error = compute_keypoint_error(pred_coords, gt_coords)
    # print(f"Keypoint RMSE error: {error:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    main(args)
