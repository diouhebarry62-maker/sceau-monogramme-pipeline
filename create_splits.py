# -*- coding: utf-8 -*-
"""
create_splits.py
----------------
Création des splits stratifiés 5-fold (train / val / test)
à partir des deux CSV annotés (Adolfo + Zacos).

Référence scikit-learn :
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# ─── À AJUSTER ────────────────────────────────────────────────────────────────
ADOLFO_CSV = "/content/bhai_classification/data/adolfo.csv"
ZACOS_CSV  = "/content/bhai_classification/data/zacos.csv"
OUTPUT_DIR = "/content/bhai_classification/splits"
N_SPLITS     = 5
VAL_SIZE     = 0.15
RANDOM_STATE = 7
# ──────────────────────────────────────────────────────────────────────────────


def load_and_merge():
    """
    Charge les deux CSV et les fusionne en un DataFrame unifié.
    Colonnes résultantes : 'image_path', 'label', 'collection'
    """
    adolfo = pd.read_csv(ADOLFO_CSV)
    zacos  = pd.read_csv(ZACOS_CSV)

    # Adolfo : monogram_id → image_path (on ajoute .jpg)
    adolfo["image_path"] = adolfo["monogram_id"].astype(str) + ".jpg"
    adolfo["collection"] = "adolfo"
    adolfo = adolfo[["image_path", "quality_label", "collection"]]

    # Zacos : filename → image_path
    zacos["image_path"] = zacos["filename"].astype(str)
    zacos["collection"] = "zacos"
    zacos = zacos[["image_path", "quality_label", "collection"]]

    df = pd.concat([adolfo, zacos], ignore_index=True)
    df = df.rename(columns={"quality_label": "label"})

    print(f"Dataset combiné : {len(df)} images")
    print(f"Distribution des labels :\n{df['label'].value_counts().sort_index()}\n")
    return df


def create_stratified_splits(output_dir=OUTPUT_DIR, n_splits=N_SPLITS):
    """
    Génère n_splits folds stratifiés (train / val / test).
    Chaque fold est sauvegardé dans splits/fold_X/ sous forme de CSV.
    """
    df = load_and_merge()
    X  = df["image_path"].values
    y  = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df_trainval = df.iloc[trainval_idx]
        df_test     = df.iloc[test_idx]

        df_train, df_val = train_test_split(
            df_trainval,
            test_size=VAL_SIZE,
            stratify=df_trainval["label"],
            random_state=RANDOM_STATE
        )

        df_train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        df_val.to_csv(  os.path.join(fold_dir, "val.csv"),   index=False)
        df_test.to_csv( os.path.join(fold_dir, "test.csv"),  index=False)

        print(f"Fold {fold} | train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

    # Vérification
    all_test = []
    for fold in range(n_splits):
        df_t = pd.read_csv(os.path.join(output_dir, f"fold_{fold}", "test.csv"))
        all_test.extend(df_t["image_path"].tolist())
    assert len(all_test) == len(set(all_test)), \
        "ERREUR : les splits de test ne sont pas mutuellement exclusifs !"
    print("\n✓ Splits créés — sets de test mutuellement exclusifs.")


if __name__ == "__main__":
    create_stratified_splits()
