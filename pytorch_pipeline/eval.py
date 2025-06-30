import torch
import matplotlib.pyplot as plt
from train import SimpleUNet, train_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

if __name__ == "__main__":
    # === Load model ===
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load("best_segmentation_model.pt"))
    model.eval()

    # === IoU Metric ===
    def iou_score(preds, targets, threshold=0.5, eps=1e-6):
        preds = (preds > threshold).float()
        targets = (targets > threshold).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
        iou = (intersection + eps) / (union + eps)
        return iou.mean().item()

    # === Evaluate IoU on training data ===
    ious = []
    with torch.no_grad():
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            iou = iou_score(preds, masks)
            ious.append(iou)

    mean_iou = sum(ious) / len(ious)
    print(f"📊 Mean IoU on training data: {mean_iou:.4f}")

    # === Visualize 3 examples ===
    imgs, masks = next(iter(train_loader))
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

    with torch.no_grad():
        preds = model(imgs)
        preds = (preds > 0.5).float()

    imgs = imgs.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    for i in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Input image
        img = imgs[i][0] if imgs[i].shape[0] == 1 else imgs[i].permute(1, 2, 0)
        axs[0].imshow(img, cmap="gray" if imgs[i].shape[0] == 1 else None)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Predicted mask
        axs[1].imshow(preds[i][0], cmap="gray")
        axs[1].set_title("Predicted Mask")
        axs[1].axis("off")

        # Ground truth mask
        axs[2].imshow(masks[i][0], cmap="gray")
        axs[2].set_title("Ground Truth Mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()
