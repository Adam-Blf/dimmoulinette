# Mode d'Emploi - Moulinettes DIM

## Guide Utilisateur pour le traitement des donnees PMSI Psychiatrie

---

## 1. Preparation des Fichiers

### Ou deposer vos fichiers ?

Placez tous vos fichiers PMSI dans le dossier :

```
C:\Users\adamb\Downloads\frer
```

> **Important** : Ne modifiez pas ce chemin. Le systeme est configure pour lire uniquement ce dossier.

### Quels fichiers sont acceptes ?

| Type | Description | Priorite |
|------|-------------|----------|
| **RPS** | Resume Par Sequence (hospitalisation psy) | Haute |
| **RAA** | Resume d'Activite Ambulatoire (consultations) | Haute |
| **RIMP** | Recueil d'Information Medicale en Psychiatrie | Haute |
| **RSS/RSA** | Autres fichiers PMSI | Moyenne |

**Formats supportes** : `.csv`, `.txt`, `.tsv`

### Comment nommer vos fichiers ?

Le systeme detecte automatiquement le type de fichier grace aux mots-cles dans le nom :
- `RPS_2024_janvier.csv` -> Detecte comme RPS
- `fichier_RAA_psy.txt` -> Detecte comme RAA
- `export_RIMP_service.csv` -> Detecte comme RPS

---

## 2. Utilisation du Dashboard

### Lancer l'application

1. Ouvrez un terminal (cmd ou PowerShell)
2. Naviguez vers le dossier du projet :
   ```
   cd C:\Users\adamb\Documents\DIM
   ```
3. Lancez le serveur :
   ```
   python app.py
   ```
4. Ouvrez votre navigateur a l'adresse : **http://localhost:8000**

### Interface principale

```
+--------------------------------------------------+
|  MOULINETTES DIM - Dashboard                     |
+--------------------------------------------------+
|                                                  |
|  [Fichiers Source]        [Actions]              |
|   - RPS: 3 fichiers        [Lancer ETL]          |
|   - RAA: 2 fichiers        [Lancer Fine-Tuning]  |
|                                                  |
|  [Console de Logs]                               |
|   [12:30:15] Pipeline ETL demarre...             |
|   [12:30:18] 5 fichiers traites                  |
|                                                  |
+--------------------------------------------------+
```

### Etape 1 : Lancer le Pipeline ETL

1. Verifiez que vos fichiers apparaissent dans la section "Fichiers Source"
2. Cliquez sur **"Lancer ETL"**
3. Attendez que le traitement se termine (suivez la console)
4. Les resultats s'affichent automatiquement

**Que fait l'ETL ?**
- Scanne et classe vos fichiers par priorite
- Fusionne les episodes de soins consecutifs
- Detecte les anomalies organisationnelles
- Genere un dataset pour l'IA

### Etape 2 : Lancer le Fine-Tuning IA

> **Prerequis** : L'ETL doit avoir ete execute au moins une fois.

1. Cliquez sur **"Lancer Fine-Tuning"**
2. L'operation peut prendre plusieurs minutes (voire heures)
3. Un spinner indique que le traitement est en cours

**Que fait le Fine-Tuning ?**
- Telecharge le modele Mistral-7B si necessaire
- Entraine le modele sur vos donnees
- Sauvegarde les adaptateurs dans `./models/adapters`

---

## 3. Interpreter les Alertes IA

### Types d'anomalies detectees

| Code | Signification | Action recommandee |
|------|---------------|-------------------|
| `ANOMALIE_ORG` | Duree anormale en ambulatoire | Verifier le codage ou l'organisation |
| `PARCOURS_ATYPIQUE` | Nombreux passages en peu de temps | Analyser le parcours patient |
| `DUREE_EXCESSIVE` | Sejour inhabituellement long | Controler les dates |

### Exemple d'alerte

```json
{
  "instruction": "Detecter une anomalie de parcours patient",
  "input": "Patient X, 5 jours consecutifs en ambulatoire SAU",
  "output": "ANOMALIE_ORG detectee: Duree excessive en ambulatoire"
}
```

**Interpretation** : Le patient a ete vu 5 jours de suite en ambulatoire, ce qui est inhabituel. Verifiez :
1. Est-ce un probleme de codage ?
2. Le patient necessite-t-il une hospitalisation ?
3. Y a-t-il un probleme organisationnel ?

---

## 4. Fichiers de Sortie

Apres le traitement, vous trouverez dans `./output/` :

| Fichier | Description |
|---------|-------------|
| `train_dataset.jsonl` | Dataset pour l'entrainement IA |
| `episodes_consolides.csv` | Tous les episodes fusionnes |

---

## 5. Securite des Donnees

### Ce que le systeme fait pour vous proteger

1. **Hash SHA-256** : Chaque fichier est verifie pour detecter les modifications
2. **Mode lecture seule** : Les fichiers source ne sont jamais modifies
3. **Aucune donnee dans Git** : Le `.gitignore` exclut toutes les donnees sensibles

### Ce que vous devez faire

- [ ] Ne jamais partager le dossier `frer` par email ou cloud non securise
- [ ] Activer BitLocker sur votre disque dur
- [ ] Fermer le dashboard quand vous ne l'utilisez pas

---

## 6. Depannage

### "Aucun fichier trouve"
- Verifiez que le dossier `C:\Users\adamb\Downloads\frer` existe
- Verifiez que les fichiers ont l'extension `.csv`, `.txt` ou `.tsv`

### "Dataset non trouve"
- Executez d'abord le pipeline ETL avant le fine-tuning

### "Erreur GPU / CUDA"
- Le fine-tuning fonctionne aussi sur CPU (plus lent)
- Verifiez que vos drivers NVIDIA sont a jour

### "Memoire insuffisante"
- Fermez les autres applications
- Le modele est optimise pour 8GB de VRAM minimum

---

## Contact Support

Pour toute question technique, consultez le fichier `README_DEV.md` ou contactez l'equipe DIM.
