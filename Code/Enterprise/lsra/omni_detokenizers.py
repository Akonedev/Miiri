import torch
import torch.nn as nn

class VisualDetokenizer(nn.Module):
    """
    Décodeur Visuel Natif (VQ-VAE Decoder).
    Transforme la séquence de tokens (Mentalese) en pixels [Batch, Channels, Height, Width].
    """
    def __init__(self, d_model=256, out_channels=3, patch_size=16, image_size=224):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        
        # ConvTranspose projette l'espace latent vers l'espace pixel
        self.deconv = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, latent_sequence):
        """
        latent_sequence: [Batch, Num_Patches, d_model]
        """
        batch_size = latent_sequence.size(0)
        # Reshape sequence into a 2D spatial grid
        x = latent_sequence.transpose(1, 2).view(batch_size, self.d_model, self.grid_size, self.grid_size)
        # Deconvolve back to pixels
        pixels = self.deconv(x)
        return torch.sigmoid(pixels) # Normalize between 0 and 1

class AudioDetokenizer(nn.Module):
    """
    Décodeur Audio Natif (Vocoder Simplifié).
    Transforme la séquence en une onde temporelle (Waveform) [Batch, 1, Length].
    """
    def __init__(self, d_model=256, upsample_factor=160):
        super().__init__()
        # ConvTranspose1D pour générer des échantillons audio à partir de frames latentes
        self.deconv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=upsample_factor,
            stride=upsample_factor
        )

    def forward(self, latent_sequence):
        # [Batch, Seq_Len, d_model] -> [Batch, d_model, Seq_Len]
        x = latent_sequence.transpose(1, 2)
        waveform = self.deconv(x)
        return torch.tanh(waveform) # Audio amplitude entre -1 et 1

class World3DDetokenizer(nn.Module):
    """
    Générateur de Monde 3D Natif.
    Transforme les entités latentes en coordonnées spatiales X, Y, Z (Point Cloud).
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.point_projector = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 3) # X, Y, Z coordinates
        )

    def forward(self, latent_sequence):
        """
        Retourne un nuage de points 3D pour chaque élément de la séquence.
        [Batch, Num_Points, 3]
        """
        points_3d = self.point_projector(latent_sequence)
        return points_3d
