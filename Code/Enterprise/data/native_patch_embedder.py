import torch
import torch.nn as nn

class NativePixelPatchEmbedder(nn.Module):
    """
    Encodeur Visuel Pur & Natif pour l'architecture Miiri.
    Zéro modèle pré-entraîné. Zéro Frankenstein.
    Prend des pixels bruts et les écrase directement dans le QPLS Mentalese (d=256).
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, d_model=256):
        super(NativePixelPatchEmbedder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        
        assert image_size % patch_size == 0, "L'image doit être divisible par la taille du patch."
        self.num_patches = (image_size // patch_size) ** 2
        
        # 1. Convolution pure : Découpe l'image en patchs et projette vers 256d
        # C'est la seule opération mathématique. Le reste sera appris par le LSRA.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. Orthogonal Sparsity Enforcer (Hardware Level)
        # On force la projection visuelle à ne remplir QUE les entités et propriétés (0 à 127).
        # Les dimensions 128-191 (Opérateurs causaux) doivent rester strictly 0.0 en entrée brute.
        self.sparsity_mask = torch.zeros(d_model)
        self.sparsity_mask[0:128] = 1.0  # Autorise Entities et Properties
        # Le masque est un paramètre non entraînable
        self.register_buffer('fixed_mask', self.sparsity_mask)
        
        # 3. Position Embedding Apprenable (Pour que l'IA comprenne la structure 2D)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

    def forward(self, pixel_tensor):
        """
        pixel_tensor : [Batch, Channels, Height, Width]
        Retourne : [Batch, Num_Patches, 256] -> Prêt pour le LSRA Core.
        """
        batch_size = pixel_tensor.shape[0]
        
        # (Batch, 256, H/16, W/16)
        x = self.proj(pixel_tensor)
        
        # Flatten: (Batch, 256, Num_Patches)
        x = x.flatten(2)
        
        # Transpose: (Batch, Num_Patches, 256)
        x = x.transpose(1, 2)
        
        # Ajouter l'information de position
        x = x + self.position_embeddings
        
        # Appliquer la Sparsité Orthogonale (Le Filtre Anti-Bruit)
        x = x * self.fixed_mask
        
        # Forcer la dimension Metadata (Incertitude initiale + Source Modale Visuelle)
        x[..., 192] = 2.0  # Identifiant de la modalité "VISION"
        
        return x
