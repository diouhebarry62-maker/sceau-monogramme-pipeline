# Lecture automatique des sceaux byzantins

Deux travaux menés dans le cadre du projet ANR **BHAI** (Hybrid Artificial Intelligence applied to Byzantine Sigillography, ANR-21-CE38-0001) à Sorbonne Université, sous la direction de Victoria Eyharabide.

| | Travail | Question | Code | Document |
|---|---|---|---|---|
| 1 | **Mémoire de M1** (soutenu le 4 juin 2026, 18/20) | Où sont les lettres dans un monogramme, et lesquelles sont-ce ? | [`1_detection_yolo/`](1_detection_yolo/) · [`2_classification_caracteres/`](2_classification_caracteres/) | [Manuscrit (PDF)](docs/memoire_M1_2026_Barry_Sawadogo.pdf) |
| 2 | **DH2026** (Daejeon, 28 juillet 2026) | Dans quel état de conservation est le sceau ? | [`3_dh2026_qualite/`](3_dh2026_qualite/) | [Slides (PDF)](docs/DH2026_slides_Byzantine_Seal_Datafication.pdf) |

**Auteurs :** Ezéchiel Sawadogo, Mamadou Diouhé Barry · **Encadrement :** Victoria Eyharabide, Saranga K. Mahanta

---

## 1. Mémoire — *Reconnaître des caractères dans des monogrammes de sceaux byzantins : un pipeline d'annotation semi-automatique*

Un monogramme byzantin superpose plusieurs lettres grecques en un seul dessin ; le même agencement de traits peut admettre plusieurs lectures. Le mémoire propose un pipeline en deux étapes :

```
image du sceau ──► YOLOv8s (détection) ──► vignettes ──► ResNet18 (classification) ──► lettres
```

| Étape | Modèle | Données | Résultat |
|---|---|---|---|
| Détection | YOLOv8s, 100 époques, 640 px | 351 images, 49 classes | mAP@50 **0,778** (5-fold) |
| Classification | ResNet18 fine-tuné (vs CNN simple, ViT-Tiny) | 2 143 vignettes, 26 classes | exactitude **96,7 %**, F1-macro 0,937 |

Détails : [`1_detection_yolo/README.md`](1_detection_yolo/README.md), [`2_classification_caracteres/README.md`](2_classification_caracteres/README.md) et le manuscrit (77 pages).

## 2. DH2026 — *Digital Curation and Datafication of Byzantine Seal Monograms: From Annotation to Classification*

Avant de lire un monogramme, il faut savoir s'il est lisible. Ce travail annote 625 sceaux de quatre collections (Adolfo-Eidelstein, Zacos, Tatış, Seyrig) selon quatre niveaux de qualité **M0–M3** (M0 : excellent état → M3 : à peine lisible) et compare plusieurs encodeurs pré-entraînés (ResNet, EfficientNet, ConvNeXt, ViT, DINOv2) en validation croisée stratifiée, avec pondération des classes pour compenser un déséquilibre fort (M3 : 29 exemples).

Détails : [`3_dh2026_qualite/README.md`](3_dh2026_qualite/README.md) et les slides.

---

## Installation

```bash
pip install -r requirements.txt
```

Le code a été écrit et exécuté sur Google Colab (GPU T4). Les chemins `BASE_DIR` / `CSV_PATH` en tête de chaque script sont à adapter.

## Données

Les images et annotations appartiennent aux collections partenaires du projet BHAI et **ne sont pas versionnées** dans ce dépôt (voir `.gitignore`). Contacter les auteurs pour toute question sur l'accès au corpus.

## Citer

```
Barry M. D., Sawadogo E. (2026). Reconnaître des caractères dans des monogrammes de sceaux byzantins :
un pipeline d'annotation semi-automatique. Mémoire de M1, Sorbonne Université, dir. V. Eyharabide.

Sawadogo E., Barry M. D., Mahanta S. K., Eyharabide V. (2026). Digital Curation and Datafication of
Byzantine Seal Monograms: From Annotation to Classification. Digital Humanities 2026, Daejeon.
```
