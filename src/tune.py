import copy
import json
import os

import mlflow
import optuna
import yaml

from train import run

cfg = yaml.safe_load(open("config.yaml"))
t = cfg["tune"]
mlflow.set_experiment(cfg["mlflow_experiment"] + "-tuning")


def objective(trial):
    c = copy.deepcopy(cfg)
    c["epochs"] = t["epochs"]
    c["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    c["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    c["rotation"] = trial.suggest_int("rotation", 0, 30, step=5)
    c["color_jitter"] = trial.suggest_float("color_jitter", 0.0, 0.4)
    return run(c, register_model=False)["val_acc"]


with mlflow.start_run(run_name="optuna"):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=t["n_trials"])
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_acc", study.best_value)

os.makedirs("reports", exist_ok=True)
json.dump({"best_val_acc": study.best_value, "best_params": study.best_params},
          open("reports/best_params.json", "w"), indent=2)
print(f"En iyi val_acc={study.best_value:.4f}")
print("En iyi ayarlar:", study.best_params)