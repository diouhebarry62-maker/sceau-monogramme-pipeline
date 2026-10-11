# -*- coding: utf-8 -*-
"""
main.py
-------
Point d'entrée : splits stratifiés, entraînement des 5 expériences sur 5 folds,
agrégation mean ± std et sauvegarde dans results/all_results.json.
"""

import os
import json
import torch
import numpy as np
import pandas as pd

from create_splits import create_stratified_splits
from utils.dataset import AdolfoDataset, ZacosDataset, ConcatDataset
from models.models import get_model
from train import train_fold

# ─── À AJUSTER ────────────────────────────────────────────────────────────────
ADOLFO_CSV    = "/content/bhai_classification/data/adolfo.csv"
ZACOS_CSV     = "/content/bhai_classification/data/zacos.csv"
ADOLFO_IMGDIR = "/content/bhai_classification/data/Adolfo/"
ZACOS_IMGDIR  = "/content/bhai_classification/data/Zacos/"
SPLITS_DIR    = "/content/bhai_classification/splits"
RESULTS_DIR   = "/content/bhai_classification/results"
N_SPLITS      = 5

EXPERIMENTS = [
    {"model": "resnet18",     "strategy": "finetune"},
    {"model": "resnet18",     "strategy": "frozen"},
    {"model": "resnet50",     "strategy": "finetune"},
    {"model": "efficientnet", "strategy": "finetune"},
    {"model": "vit_tiny",     "strategy": "finetune"},
]
# ──────────────────────────────────────────────────────────────────────────────

def load_split_datasets(fold, split):
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"fold_{fold}", f"{split}.csv"))
    df_adolfo = df[df["collection"] == "adolfo"].reset_index(drop=True)
    df_zacos  = df[df["collection"] == "zacos"].reset_index(drop=True)

    tmp_a = f"/tmp/adolfo_{split}_{fold}.csv"
    tmp_z = f"/tmp/zacos_{split}_{fold}.csv"

    df_adolfo[["image_path", "label"]].rename(
        columns={"image_path": "monogram_id", "label": "quality_label"}
    ).assign(
        monogram_id=lambda x: x["monogram_id"].str.replace(".jpg", "", regex=False)
    ).to_csv(tmp_a, index=False)

    df_zacos[["image_path", "label"]].rename(
        columns={"image_path": "filename", "label": "quality_label"}
    ).to_csv(tmp_z, index=False)

    adolfo_ds = AdolfoDataset(tmp_a, ADOLFO_IMGDIR, split)
    zacos_ds  = ZacosDataset(tmp_z, ZACOS_IMGDIR,  split)
    return ConcatDataset([adolfo_ds, zacos_ds])

def run_experiment(model_name, strategy, device):
    print(f"\n{'='*60}")
    print(f"  Expérience : {model_name} | stratégie : {strategy}")
    print(f"{'='*60}")
    fold_results = []
    for fold in range(N_SPLITS):
        train_ds = load_split_datasets(fold, "train")
        val_ds   = load_split_datasets(fold, "val")
        test_ds  = load_split_datasets(fold, "test")
        model    = get_model(model_name, strategy)
        metrics  = train_fold(model, train_ds, val_ds, test_ds, device, fold)
        fold_results.append(metrics)
    return fold_results

def aggregate_results(fold_results, model_name, strategy):
    test_accs = [r["test_acc"] for r in fold_results]
    test_f1s  = [r["test_f1"]  for r in fold_results]
    best_fold = int(np.argmax(test_accs))
    summary = {
        "model"         : model_name,
        "strategy"      : strategy,
        "mean_test_acc" : float(np.mean(test_accs)),
        "std_test_acc"  : float(np.std(test_accs)),
        "mean_test_f1"  : float(np.mean(test_f1s)),
        "std_test_f1"   : float(np.std(test_f1s)),
        "best_fold"     : best_fold,
        "best_test_acc" : float(test_accs[best_fold]),
        "fold_details"  : fold_results
    }
    print(f"\n  ► {model_name} ({strategy})")
    print(f"    test_acc : {summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}")
    print(f"    test_f1  : {summary['mean_test_f1']:.4f} ± {summary['std_test_f1']:.4f}")
    print(f"    meilleur fold : {best_fold} (acc={summary['best_test_acc']:.4f})")
    return summary

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SPLITS_DIR):
        print("\n[1/3] Création des splits stratifiés...")
        create_stratified_splits()
    else:
        print("\n[1/3] Splits déjà existants — on passe.")

    print("\n[2/3] Entraînement des modèles...")
    all_summaries = []
    for exp in EXPERIMENTS:
        fold_results = run_experiment(exp["model"], exp["strategy"], device)
        summary      = aggregate_results(fold_results, exp["model"], exp["strategy"])
        all_summaries.append(summary)

    print("\n[3/3] Sauvegarde des résultats...")
    out_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Résultats sauvegardés dans {out_path}")
    print("\n─── Récapitulatif final ───────────────────────────────────")
    for s in all_summaries:
        print(f"  {s['model']:15s} ({s['strategy']:8s}) | "
              f"acc={s['mean_test_acc']:.4f}±{s['std_test_acc']:.4f} | "
              f"f1={s['mean_test_f1']:.4f}±{s['std_test_f1']:.4f} | "
              f"best_fold={s['best_fold']}")

if __name__ == "__main__":
    main()
