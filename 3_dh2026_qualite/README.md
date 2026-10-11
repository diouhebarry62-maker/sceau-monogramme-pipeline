# DH2026 — Classification de la qualité des sceaux byzantins (M0–M3)

*Digital Curation and Datafication of Byzantine Seal Monograms: From Annotation to Classification* — présenté à Digital Humanities 2026 (Daejeon), mini-conférence « Distant Past(s) – Current Future(s) ». Slides : [`../docs/DH2026_slides_Byzantine_Seal_Datafication.pdf`](../docs/DH2026_slides_Byzantine_Seal_Datafication.pdf).

Pipeline de classification multiclasse par transfer learning : 625 sceaux de quatre collections, annotés en quatre niveaux de qualité.

| Classe | Interprétation | Effectif |
|---|---|---|
| M0 | Excellent état, bon contraste | 80 |
| M1 | Bon état, défauts mineurs | 272 |
| M2 | Dégradé, composants cassés, lisibilité partielle | 244 |
| M3 | Très dégradé, monogramme à peine lisible | 29 |

**État du code.** Scripts `create_splits.py` et `main.py` présents ; `train.py`, `models/models.py`, `utils/dataset.py` et `visualize_results.py` restent à ajouter. Deux notebooks Colab :
- `quality_classification_5fold.ipynb` — version de référence : splits stratifiés 5 folds, pondération des classes et sampler pondéré, comparaison ResNet-18 / ResNet-50 / EfficientNet / ViT-Tiny, résultats mean ± std (c'est le pipeline décrit ci-dessous).
- `quality_classification.ipynb` — variante exploratoire (fusion des annotations Zacos / Adolfo-Eidelstein, dataset PyTorch, pondération des classes, early stopping, comparaison EfficientNet-B0 / ConvNeXt-Tiny / ViT-B/16 / DINOv2-Small, matrices de confusion). La structure modulaire ci-dessous (`main.py`, `create_splits.py`…) est la cible de refactorisation.

**Auteurs :** Ezéchiel Sawadogo & Mamadou Diouhé Barry  
**Encadrement :** Victoria Eyharabide & Saranga K. Mahanta

---

## Structure du projet

```
bhai_classification/
│
├── main.py                  # ← point d'entrée principal
├── create_splits.py         # création des splits stratifiés (StratifiedKFold)
├── train.py                 # boucle d'entraînement et d'évaluation par fold
├── visualize_results.py     # graphiques (barplot, matrices de confusion)
├── requirements.txt
│
├── data/
│   ├── dataset.csv          # ← À FOURNIR par Saranga (colonnes: image_path, label)
│   └── images/              # ← À FOURNIR : dossier des ~800 images de sceaux
│
├── models/
│   └── models.py            # ResNet18, ResNet50, EfficientNet-B0, ViT-Tiny
│
├── utils/
│   └── dataset.py           # Dataset PyTorch + transformations
│
├── splits/                  # généré automatiquement par create_splits.py
│   ├── fold_0/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── ...
│
└── results/                 # généré automatiquement
    ├── all_results.json     # métriques mean ± std de tous les modèles
    ├── best_model_fold*.pth # checkpoints des meilleurs modèles
    └── figures/             # barplots et matrices de confusion
```

---

## Format du CSV attendu

Le fichier `data/dataset.csv` doit avoir exactement ces deux colonnes :

```
image_path,label
images/seal_001.jpg,M0
images/seal_002.jpg,M2
...
```

Labels : `M0`, `M1`, `M2`, `M3`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Placer les données
- Copier `dataset.csv` dans `data/`
- Copier les images dans `data/images/`

### 2. Lancer le pipeline complet
```bash
python main.py
```

Cela va :
1. Créer les splits stratifiés (5-fold)
2. Entraîner chaque modèle sur chaque fold
3. Afficher et sauvegarder les résultats (mean ± std)

### 3. Générer les visualisations
```bash
python visualize_results.py
```

---

## Modèles et stratégies comparées

| Modèle        | Stratégie  | Description                        |
|---------------|------------|------------------------------------|
| ResNet-18     | finetune   | Fine-tuning complet                |
| ResNet-18     | frozen     | Backbone gelé, tête seule          |
| ResNet-50     | finetune   | Fine-tuning complet                |
| EfficientNet  | finetune   | Fine-tuning complet                |
| ViT-Tiny      | finetune   | Vision Transformer (attention)     |

---

## Résultats attendus (format de sortie)

```
resnet18        (finetune) | acc=0.XXXX±0.XXXX | f1=0.XXXX±0.XXXX | best_fold=X
resnet18        (frozen  ) | acc=0.XXXX±0.XXXX | f1=0.XXXX±0.XXXX | best_fold=X
...
```

---

## À AJUSTER avant de lancer

Dans `main.py` :
- `CSV_PATH` → chemin vers le CSV de Saranga
- `IMG_DIR`  → chemin vers le dossier des images

Dans `train.py` :
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE` selon les ressources disponibles (Colab T4)
