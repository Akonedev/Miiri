# Chapitre 11 : Équations Fondamentales et Preuves Mathématiques
*Ce chapitre compile l'intégralité des formules mathématiques régissant l'architecture Miiri, de la mécanique des tenseurs à la modélisation différentielle de la mémoire.*

## 1. Topologie Tensorielle du Mentalese (QPLS)

L'espace latent $v \in \mathbb{R}^{d}$ (avec $d=256$) n'est pas isotrope. Il est défini comme la concaténation directe de quatre sous-espaces orthogonaux :

$$ \mathbf{v}^{(t)} = \left[ \mathbf{v}_{ent}^{(t)} \parallel \mathbf{v}_{prop}^{(t)} \parallel \mathbf{v}_{op}^{(t)} \parallel \mathbf{v}_{meta}^{(t)} \right] $$

Où chaque $\mathbf{v}_{k} \in \mathbb{R}^{64}$. L'orthogonalité sémantique est garantie par la contrainte de produit scalaire nul entre les sous-espaces lors de l'initialisation des primitives :
$$ \forall i \neq j, \quad \langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0 $$

## 2. Fonction de Perte ACSP (Amodal Consistency & Step Penalty)

La perte globale qui empêche l'hallucination et force l'apprentissage causal est définie par :

$$ \mathcal{L}_{ACSP} = \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{step} + \lambda_3 \mathcal{L}_{sparse} + \lambda_4 \mathcal{L}_{consist} $$

### A. Pénalité de Rigueur Causale ($\mathcal{L}_{step}$)
Soit $\mathbb{V}: \mathbb{R}^{64} \times \mathbb{R}^{64} \times \mathbb{R}^{64} \rightarrow \{0, 1\}$ la fonction indicatrice du Moteur Symbolique. Elle renvoie $1$ si la transition est légale, $0$ sinon. La pénalité s'écrit avec une fonction de Dirac modifiée :

$$ \mathcal{L}_{step} = \sum_{t=1}^{T} \mathbb{1}_{\{\mathbb{V}(\mathbf{v}_{ent}^{(t)}, \mathbf{v}_{prop}^{(t)}, \mathbf{v}_{op}^{(t)}) = 0\}} \cdot P_{backtrack} $$
*(Avec $P_{backtrack} = 1000.0$. Cela crée une discontinuité massive dans le paysage des gradients, forçant l'optimiseur AdamW à fuir cette région de l'espace latent).*

### B. Consistance Amodale (InfoNCE Modifiée)
Pour un concept $C$ perçu par les modalités Texte ($T$) et Vision ($V$) :

$$ \mathcal{L}_{consist} = - \mathbb{E}_{(x_T, x_V) \sim C} \left[ \log \frac{\exp(\text{sim}(f_T(x_T), f_V(x_V)) / \tau)}{\sum_{x_j \in Batch} \exp(\text{sim}(f_T(x_T), f_V(x_j)) / \tau)} \right] $$
Où $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ est la similarité cosinus.

## 3. Dynamique de Récurrence Latente (LSRA)

La mise à jour de l'état caché dans le *Reasoning Core* est définie par :

$$ \mathbf{v}^{(t+1)} = \text{LayerNorm}\left( \mathbf{v}^{(t)} + \sigma_{SN}\left( \text{FFN}(\mathbf{v}^{(t)}) \right) \right) $$

Où $\sigma_{SN}$ représente la Normalisation Spectrale de la matrice de poids $W$, garantissant que le rayon spectral $\rho(W) < 1$. C'est cette condition mathématique (Théorème du point fixe de Banach) qui permet au modèle de boucler à l'infini sans explosion de la norme $\| \mathbf{v}^{(t)} \|_2$.

## 4. Modélisation de la Mémoire (Analogie du Circuit LC Résonant)

Pour prouver que l'information survit à 10 000 itérations, nous modélisons l'état caché $\mathbf{h}^{(t)}$ du Transformer comme un système dynamique différentiel du second ordre (Équation de l'oscillateur harmonique amorti) :

$$ \frac{d^2\mathbf{h}}{dt^2} + 2\zeta\omega_n \frac{d\mathbf{h}}{dt} + \omega_n^2 \mathbf{h} = \mathbf{x}(t) $$

Dans l'espace discret de Miiri, la matrice d'attention agit comme un filtre passe-bande. Si les valeurs propres ($\lambda$) de la matrice de transition récurrente sont des conjugués complexes dont le module est très proche de 1 ($|\lambda| \approx 1 - \epsilon$), le système a un **facteur de qualité $Q$ extrêmement élevé** :

$$ Q = \frac{\omega_n}{2\zeta} \gg 1 $$

**Conclusion Mathématique :** L'identité du vecteur (sa signification) oscille dans l'espace latent sans s'amortir. L'information n'est pas "oubliée" par le réseau, elle entre en résonance.
