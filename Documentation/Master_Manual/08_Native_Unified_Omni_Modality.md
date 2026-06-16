# Chapitre 8 : Native Unified Omni-Modality (Zéro "Frankenstein")

*Axiome fondamental : L'Miiri-256 n'est pas un assemblage de modèles. Il est UNIFIÉ de bout en bout.*

## 8.1 L'Anti-Pattern "Frankenstein"
La majorité des architectures prétendument multimodales de 2023-2024 utilisent une approche "Late Fusion" (Frankenstein). Elles utilisent un LLM pour le texte, puis l'utilisent comme "prompteur" pour piloter un modèle externe (comme Stable Diffusion ou Midjourney) pour générer l'image.
**Pourquoi c'est inacceptable pour Miiri-256 :**
1. **Perte de Rigueur Causale :** Si le LLM demande à Stable Diffusion de générer "une pomme qui tombe à 9.8m/s²", le modèle de diffusion ne *comprend* pas la physique. Il va halluciner des pixels qui "ressemblent" à une chute.
2. **Latence et Déconnexion :** Le passage par des modèles externes détruit le flux du *Test-Time Compute*. L'espace latent n'est plus partagé.

## 8.2 La Solution : L'Omni-Tokenizer Universel
Pour que le modèle soit UNIFIÉ, la perception (In) et la génération (Out) de toutes les modalités doivent partager le **MÊME vocabulaire discret**.

*   **Texte :** Mots découpés en sous-mots (Byte-Pair Encoding).
*   **Vision (Images/Vidéos) :** Images découpées en "Patchs" via un VQ-VAE (Vector Quantized Variational AutoEncoder). Un "Patch" devient un "Token Visuel" (un identifiant entier, exactement comme un mot).
*   **Audio :** Son découpé en trames de spectrogrammes, converties en "Tokens Audio".

Tous ces tokens (ex: un vocabulaire total de 200 000 éléments où les ID 0-100k sont des mots, 100k-150k sont des patchs d'images, 150k-200k sont des sons) sont projetés dans le **MÊME espace Mentalese (QPLS, $d=256$)**.

## 8.3 Le Native Omni-Decoder (Génération)
Le modèle n'a pas besoin de Sora ou de Stable Diffusion. 
À la sortie de la boucle de raisonnement (Ports 26415-26418), le modèle produit une séquence de vecteurs latents (Mentalese).

Le **Native Omni-Decoder** prend ces vecteurs et les projette sur la distribution des probabilités du vocabulaire unifié. 
*   S'il veut parler, il choisit les ID de tokens textuels.
*   S'il veut "montrer", il choisit les ID de tokens visuels. 
Ces tokens visuels sont ensuite décompressés par le VQ-VAE en pixels bruts. 

**Résultat :** L'image générée est la traduction *directe et mathématique* de la pensée causale du modèle. Si le moteur symbolique a validé la trajectoire de la chute dans l'espace latent, les pixels générés respecteront **strictement** les lois de la physique. Le modèle est 100% UNIFIÉ.
