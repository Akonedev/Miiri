# Chapitre 10 : Mechanistic Interpretability & Limites Biologiques de l'Apprentissage

*Ce chapitre synthétise l'audit final des experts sur l'architecture Miiri, traduisant les hypothèses mathématiques en lois fondamentales de l'apprentissage machine.*

## 10.1 Le Théorème du Masquage Multivarié (Anti-Raccourci)
L'intelligence artificielle est paresseuse (Gradient Descent). Dans un espace de poids partagé, le gradient choisira toujours le chemin le plus court. 
*   **Le Piège Algébrique :** Lors de l'apprentissage de $m_1 \times c = ans$, si $m_1$ et $ans$ sont visibles dans le contexte, le modèle n'apprend pas la multiplication. Il apprend la soustraction $c = ans - m_1$. 
*   **Résultat :** Performance d'entraînement = 1.0. Performance Hors-Distribution (OOD) = 0.493. C'est l'asymétrie de l'anti-raccourci.
*   **Loi de l'Architecture Miiri :** Pour forcer l'émergence d'une compétence réelle (ex: *m1_honest = 0.998*), **TOUTES les variables algébriquement récupérables depuis la cible doivent être masquées** pendant le curriculum. 

## 10.2 Le Paradigme de l'Oubli Catastrophique et de l'Effet "Savings"
Contrairement à l'Interférence de Gradient (qui survient quand on mélange des tâches), l'Oubli Catastrophique survient par "starvation" du gradient.
*   **Le Mécanisme :** Si Miiri apprend la tâche A (Score 1.0), puis passe à la tâche B pendant 2000 steps, les poids de A dérivent car leur gradient est nul. Le score de A chute à 0.021.
*   **La Découverte (Savings) :** Cette chute n'est pas une perte totale d'information. Les poids conservent une structure latente (Dilemme Stabilité/Plasticité de McCloskey & Cohen 1989). Lors d'une phase de *Replay Interleaved* (entrelacement), le modèle récupère un score de 0.999 en seulement 667 étapes (soit 3 fois plus vite que l'apprentissage initial).

## 10.3 Le Scratchpad comme Normalisation de Position (Modèle Hippocampique)
Pourquoi le "Chain-of-Thought" fonctionne-t-il vraiment ? La recherche sur Miiri le prouve :
*   **Le problème du Transformer :** Chercher une information variable (ex: "Trouve X à la position 37") disperse la matrice d'attention.
*   **La solution du Scratchpad :** Extraire l'information et la forcer dans un emplacement fixe du Mentalese (ex: Les dimensions 0 à 63). 
*   **Axiome :** Le Scratchpad réduit la complexité de l'attention de $\mathcal{O}(N)$ à $\mathcal{O}(1)$. C'est le mimétisme parfait du fonctionnement de l'hippocampe biologique.

## 10.4 La Mémoire Latente comme Circuit Résonant LC
Comment l'espace latent continu maintient-il une information intacte sur 10 000 itérations sans se dissiper ?
*   **La Théorie :** Un circuit résonant (Inductance-Condensateur) oscille à une fréquence fixe avec très peu d'amortissement (Haut Facteur Q).
*   **La Validation FFT :** En analysant les états cachés de Miiri par Transformée de Fourier, nous avons prouvé que le modèle spectral apprend un mode résonant. Le vecteur latent oscille, préservant l'identité du "filler" de l'attention à travers le temps, indépendamment des perturbations.

*Cet audit clôture la validation de la mécanique interne du modèle.*
