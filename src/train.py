import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    size = (cfg["img_size"], cfg["img_size"])
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(cfg["rotation"]),
        transforms.ColorJitter(cfg["color_jitter"], cfg["color_jitter"], cfg["color_jitter"]),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), norm])

    def loader(split, tf, shuffle):
        ds = datasets.ImageFolder(f"{cfg['processed_dir']}/{split}", tf)
        return ds, DataLoader(ds, cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"])

    train_ds, train_dl = loader("train", train_tf, True)
    _, val_dl = loader("val", eval_tf, False)
    _, test_dl = loader("test", eval_tf, False)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(train_ds.classes))
    model.to(dev)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    def evaluate(dl):
        model.eval()
        loss = correct = n = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                loss += loss_fn(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                n += len(y)
        return loss / n, correct / n

    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in cfg.items() if k != "classes"})
        mlflow.log_param("classes", train_ds.classes)
        mlflow.log_param("device", str(dev))
        for ep in range(cfg["epochs"]):
            print(f"epoch {ep + 1}/{cfg['epochs']} basladi", flush=True)
            model.train()
            for x, y in train_dl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
            vl, va = evaluate(val_dl)
            mlflow.log_metrics({"val_loss": vl, "val_acc": va}, step=ep)
            print(f"epoch {ep + 1}/{cfg['epochs']} bitti val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        _, ta = evaluate(test_dl)
        mlflow.log_metric("test_acc", ta)
        print(f"test_acc={ta:.4f}")
        model.cpu()
        os.makedirs("models", exist_ok=True)
        torch.save(model, "models/model.pt")
        try:
            example = np.zeros((1, 3, cfg["img_size"], cfg["img_size"]), dtype=np.float32)
            mlflow.pytorch.log_model(model, "model", input_example=example)
        except Exception as e:
            print(f"UYARI: mlflow log_model basarisiz: {e}")


if __name__ == "__main__":
    main()
