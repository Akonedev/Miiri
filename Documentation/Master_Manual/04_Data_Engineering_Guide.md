# Chapitre 4 : Data Engineering & Curriculum (Production Grade)

L'OCM-26400 nécessite un pipeline de données radicalement différent de celui d'un LLM de type GPT. Vous n'entraînez pas sur du texte brut extrait du web. Vous construisez un **Cursus de Primitives**.

## 4.1 Formatage des Données (Orthogonal Sparsity)
Les données doivent être ingérées via des formats Big Data (Parquet ou JSONL).
Le DataLoader doit forcer la *Sparsité Orthogonale* : 
- Lors de la phase de Grokking, si la donnée est identifiée comme une `ENTITY`, le Data Engineer **doit** blinder de zéros les segments `Properties` et `Operators` du vecteur QPLS avant l'envoi au GPU.
- **Le Garde-Fou (Collate_fn) :** Le code PyTorch intègre un `assert sum(active_blocks) == 1`. Si un vecteur d'entraînement est "emmêlé", le programme plante sciemment pour empêcher de polluer la mémoire sémantique du modèle.

## 4.2 Le Curriculum d'Apprentissage (En deux passes)

### Passe 1 : Apprentissage des Bases (Primitives)
Le dictionnaire complet est injecté (Grammaire, vocabulaire, règles de physique).
- **Règle absolue :** Ne jamais passer à la Passe 2 tant que la Loss n'a pas atteint le `grokking_threshold` (ex: 0.99) sur l'intégralité du dictionnaire. 

### Passe 2 : Entraînement par Expérience (Décomposition)
Le modèle reçoit des jeux de données d'exercices.
- **Donnée d'entrée :** Un problème complexe.
- **Label (Truth) :** Non pas la réponse finale, mais la **trace exacte des étapes du Scratchpad**. Le modèle apprend à faire des liens, à utiliser ce qu'il a appris, en s'exerçant. La Loss ACSP corrige la trajectoire si le modèle tente de sauter des étapes.

## 4.3 Pipeline de Data Loading PyTorch
Utiliser des `IterableDataset` avec `pin_memory=True` pour garantir des transferts PCIe optimaux. Éviter toute sérialisation de dictionnaires Python ; le DataLoader ne doit yield que des Tenseurs purs.
