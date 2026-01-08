# Guide Utilisateur - DIM

## Introduction

DIM (Data Intelligence Médicale) est un outil de traitement des données PMSI spécialement conçu pour la psychiatrie. Il permet de :

- Transformer les fichiers bruts PMSI en fichiers CSV exploitables
- Reconstruire les parcours patients (épisodes de soins)
- Détecter les anomalies organisationnelles
- Visualiser la structure GHT
- Entraîner une IA pour prédire les anomalies

## Démarrage rapide

### 1. Lancer l'interface

Double-cliquez sur le fichier `lancer_dim.bat` ou exécutez :

```
python app.py
```

L'interface s'ouvre automatiquement dans votre navigateur à l'adresse : **http://localhost:8000**

### 2. Interface principale

L'interface se compose de plusieurs zones :

| Zone | Description |
|------|-------------|
| **Fichiers Source** | Liste des fichiers PMSI détectés dans le dossier source |
| **Épisodes PSY** | Génération des parcours ambulatoires |
| **Structure GHT** | Visualisation de l'organisation |
| **Console** | Affichage des logs en temps réel |
| **Résultats** | Tableaux des résultats et anomalies |

## Fonctionnalités

### A. Traitement des fichiers (Moulinettes)

#### Étape 1 : Vérifier les fichiers
Dans la zone "Fichiers Source", vous voyez :
- Le compteur RPS (fichiers hospitalisation psychiatrie)
- Le compteur RAA (fichiers ambulatoire psychiatrie)
- La liste complète des fichiers

#### Étape 2 : Sélectionner et lancer
1. Cochez les fichiers à traiter (ou laissez vide pour tout traiter)
2. Cliquez sur **"Lancer Moulinettes"**
3. Suivez la progression dans la console

#### Résultats
Les fichiers CSV sont générés dans le dossier `output/` :
- `fichier_rps.csv` : données structurées
- `train_dataset.jsonl` : données pour l'IA

---

### B. Génération des Épisodes Ambulatoires

Le format national RAA ne permet pas de suivre les parcours patients. DIM reconstruit les "épisodes de soins" en regroupant les actes consécutifs.

#### Paramètres

| Paramètre | Description | Valeur conseillée |
|-----------|-------------|-------------------|
| **Fichier RAA** | Fichier source à analyser | Sélectionner dans la liste |
| **Seuil anomalie** | Durée max normale (jours) | 1 jour |

#### Interprétation

- **Épisode** : Suite d'actes consécutifs pour un même patient/UM/site
- **Anomalie** : Épisode dont la durée dépasse le seuil
  - Cause possible : problème d'aval, codage incorrect, désorganisation

#### Fichiers générés
- `fichier_episodes.csv` : données enrichies avec ID épisode
- `fichier_anomalies.csv` : liste des anomalies
- `fichier_synthese.txt` : rapport de synthèse

---

### C. Visualisation Structure GHT

Transforme les fichiers Excel de structure (souvent illisibles) en graphe visuel.

#### Utilisation
1. Sélectionnez le fichier Excel de structure
2. Choisissez le format de sortie (PNG recommandé)
3. Cliquez sur **"Générer Graphe"**

#### Fichiers générés
- `structure_ght.png` : Image du graphe hiérarchique
- `structure_ght.html` : Version interactive
- `structure_ght_pmsi_pilot.json` : Format injectable dans PMSI-Pilot

---

### D. Fine-Tuning IA

Entraîne un modèle d'IA (Mistral-7B) sur vos données pour détecter automatiquement les anomalies.

#### Prérequis
- Carte graphique NVIDIA avec 12+ GB de VRAM
- Données d'entraînement générées (étape A)

#### Lancement
1. Vérifiez que le GPU est détecté (affiché dans l'interface)
2. Définissez le nombre d'epochs (3 recommandé)
3. Cliquez sur **"Fine-Tuner l'IA"**

⚠️ **Attention** : L'entraînement peut durer plusieurs heures.

---

## Lecture des résultats

### Tableau des anomalies

| Colonne | Signification |
|---------|---------------|
| **Episode ID** | Identifiant unique de l'épisode |
| **Patient** | Numéro patient anonymisé |
| **Durée (j)** | Durée de l'épisode en jours |
| **Anomalie** | Type d'anomalie détectée |

### Types d'anomalies

| Code | Description | Action recommandée |
|------|-------------|-------------------|
| `DUREE_EXCESSIVE` | Séjour trop long en ambulatoire | Vérifier codage ou problème d'aval |
| `CHEVAUCHEMENT` | Dates incohérentes | Vérifier les dates de saisie |
| `MULTI_SITE` | Patient sur plusieurs sites | Vérifier le parcours patient |

---

## Bonnes pratiques

### Sécurité des données

1. **Ne jamais copier** les données vers un cloud ou email
2. **Vérifier le .gitignore** avant tout commit Git
3. Les données restent **100% locales**

### Workflow recommandé

1. Déposer les fichiers PMSI dans le dossier source
2. Lancer les moulinettes ETL
3. Générer les épisodes pour les fichiers RAA
4. Analyser les anomalies
5. Exporter les résultats pour le DIM

### Formats de fichiers acceptés

| Extension | Type |
|-----------|------|
| `.txt` | Fichiers positionnels PMSI |
| `.csv` | Fichiers délimités |
| `.xlsx` | Fichiers Excel (structure GHT) |

---

## Dépannage

### "Dossier source non trouvé"

Le dossier `C:\Users\adamb\Downloads\frer` n'existe pas.
→ Créez le dossier et déposez-y vos fichiers PMSI.

### "Aucun fichier détecté"

Les fichiers doivent avoir l'extension `.txt`, `.csv` ou `.xlsx`.
→ Vérifiez les extensions de vos fichiers.

### "GPU non détecté"

PyTorch ne trouve pas de carte graphique NVIDIA.
→ Installez les drivers CUDA : https://developer.nvidia.com/cuda-downloads
→ Réinstallez PyTorch avec support CUDA

### "Erreur de parsing"

Le format du fichier n'est pas reconnu.
→ Vérifiez que le fichier est au format PMSI standard
→ Contactez le support pour ajouter un nouveau format

---

## Contact & Support

Pour toute question ou demande d'évolution, contacter le DIM.

---

*Document mis à jour : 2024*
