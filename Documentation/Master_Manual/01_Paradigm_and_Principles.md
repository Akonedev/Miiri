# Chapitre 1 : Le Paradigme Neuro-Symbolique et les Principes Fondamentaux

## 1.1 La Limite du "Next-Token Prediction" (Système 1)
Les LLMs contemporains (Generative Pre-trained Transformers) reposent sur un paradigme statistique de prédiction du prochain mot. Bien que performants pour le mimétisme linguistique, ils échouent fondamentalement sur la **généralisation compositionnelle systématique** (OOD - Out of Distribution). Lorsqu'ils font des mathématiques ou de la physique, ils "devinent" le résultat le plus probable statistiquement, ce qui conduit inévitablement aux **hallucinations**. 
L'approche Miiri passe au **Système 2 (Pensée Lente, Délibérative et Vérifiée)**.

## 1.2 Le Langage de la Pensée (Mentalese)
Théorisé par Jerry Fodor (1975), le Mentalese stipule que la pensée ne se fait pas en "Français" ou en "Anglais", mais dans un espace conceptuel abstrait et amodal.
L'Miiri modélise ce Mentalese sous la forme d'un vecteur structuré de dimension 256 ($d=256$).
**Avantage critique :** La compositionnalité. Dans cet espace, `COMPOSE(Chat, Noir)` n'est pas une concaténation de texte, mais une opération tensorielle vérifiable, indépendante de la modalité d'entrée (texte, son, image).

## 1.3 Grokking vs Scaling Laws
Le paradigme Miiri rejette la "Scaling Law" brute (plus de paramètres = plus intelligent). La recherche (2024-2025) prouve que l'augmentation des paramètres pour apprendre des tâches complexes favorise la mémorisation de "raccourcis" statistiques et retarde le **Grokking** (l'effondrement de la perte de validation lorsque le modèle "comprend" la règle sous-jacente).
**Le principe Miiri :** Pré-entraîner le modèle **exclusivement sur des primitives** (dictionnaire morphologique, lois de base) jusqu'à ce qu'elles soient "grokkées" à 100%. La complexité n'est ensuite atteinte que par la *décomposition explicite*, pas par l'augmentation de la taille du réseau.

## 1.4 Test-Time Compute (TTC) et Récurrence Fenêtrée
Pour résoudre un problème très difficile, un humain réfléchit plus longtemps, il ne fait pas grossir son cerveau.
L'Miiri utilise la **Récurrence Latente (LSRA)** : un bloc neuronal restreint qui boucle sur son propre état caché. 
*   **Découplage :** La "profondeur de raisonnement" est totalement découplée du nombre de paramètres et de la longueur de la fenêtre de contexte.
*   Le modèle peut boucler 10, 100 ou 1000 fois dans son espace latent interne avant d'émettre le moindre mot, rendant le raisonnement asynchrone par rapport à la génération de sortie.
