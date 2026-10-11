# Étape 2 — Classification des caractères

Identifier la lettre contenue dans chaque vignette de caractère découpée à partir des annotations (boîtes) de l'étape 1.

## Données

- 351 fichiers d'annotation (un par monogramme, format JSON Label Studio) → **2 143 vignettes** après nettoyage
- **26 classes** : Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Ϲ Τ Φ Χ Ψ ω, V, ligature OU
- Nettoyage : fusion des variantes de sigma (`C`, `C moon-shaped sigma` → `Ϲ`), suppression des classes résiduelles (< 10 exemples)
- Split stratifié 80 / 20 : 1 724 vignettes d'entraînement, 419 de test

## Modèles comparés

| Modèle | Entrée | Époques | Exactitude test |
|---|---|---|---|
| CNN simple (3 blocs conv) | 64 px | 50 | ≈ 0,94 |
| **ResNet18** pré-entraîné ImageNet, dernières couches fine-tunées (75 % des paramètres) | 224 px | 40 | **0,967** |
| ViT-Tiny (timm, `vit_tiny_patch16_224`) | 224 px | 40 | ≈ 0,95 |

Rapport de classification par lettre, matrices de confusion et courbes d'apprentissage dans le notebook ; F1-macro ResNet18 : 0,937. Les confusions restantes concernent les classes rares (Ζ, Ξ, Ψ) et les paires visuellement proches.

## Fichier

`character_classification.ipynb` — notebook complet : extraction des vignettes, nettoyage des classes, split, entraînement des trois modèles, évaluation, sauvegarde des résultats (`results_*.json`) et des poids (`best_*.pth`, non versionnés).

Chemin `SPLIT_DIR` à adapter en tête des cellules (exécuté en local, CPU).
