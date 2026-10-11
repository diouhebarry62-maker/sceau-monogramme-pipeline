# Étape 1 — Détection des caractères (YOLOv8s)

Localiser chaque caractère grec dans l'image d'un monogramme, sous forme de boîtes englobantes.

## Données

- 351 images de monogrammes, annotées manuellement (Label Studio, format YOLO)
- 49 classes de caractères
- Splits en 5 folds fournis dans `splits/fold_{0..4}/{train,test}.txt` (non versionnés)

Arborescence attendue sous `BASE_DIR` :

```
368651_Byzantine_monograms/
├── images/train/        # .png / .jpg
├── labels/train/        # .txt YOLO (class x_center y_center w h)
├── splits/fold_0/{train,test}.txt
├── data_config.yaml     # nc, names
└── runs/                # généré : fold_k/weights/best.pt
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `train.py` | Entraîne un YOLOv8s par fold (100 époques, 640 px, batch 16). Construit un dossier temporaire par fold à partir des listes de splits, génère le YAML, lance `model.train`, nettoie. |
| `eval.ipynb` | Recharge `best.pt` de chaque fold, évalue sur le split test, agrège mAP@50 / mAP@50-95 / précision / rappel (moyenne ± écart-type). |
| `predict.ipynb` | Applique le meilleur modèle à de nouvelles images : images annotées + fichiers `.txt` YOLO, utilisables comme pré-annotation à corriger (d'où le « semi-automatique »). |

## Résultats

| Métrique (5-fold) | Valeur |
|---|---|
| mAP@50 | **0,778** |

Analyse des erreurs par classe dans le manuscrit (chap. résultats).

## Lancer

```bash
python train.py          # adapter BASE_DIR en tête de fichier
```

Sur Colab : décommenter `!pip install ultralytics` et le montage Drive.
