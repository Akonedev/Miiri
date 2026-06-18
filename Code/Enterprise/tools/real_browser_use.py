import urllib.request
from bs4 import BeautifulSoup
import torch

class RealBrowserTool:
    """
    Implémentation réelle de la compétence 'Browser User' (Web Scraping).
    Ce n'est pas un mock. Il télécharge et parse de vraies pages web.
    """
    def __init__(self, d_model=256):
        self.d_model = d_model
        
    def navigate_and_read(self, url):
        """
        Navigue vers l'URL, télécharge le HTML, extrait le texte pur,
        et le prépare pour l'encodage Mentalese.
        """
        print(f"[ACTION_BROWSER] Navigation réelle vers : {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode('utf-8')
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Suppression des scripts et styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            # Ne garder que les 1000 premiers caractères pour le prototype rapide
            clean_text = text[:1000] 
            
            print(f"[ACTION_BROWSER] Succès. {len(clean_text)} caractères extraits.")
            return clean_text
            
        except Exception as e:
            print(f"[ACTION_BROWSER] Erreur de navigation : {e}")
            return None

    def text_to_naive_qpls(self, text):
        """
        Génère un véritable tenseur PyTorch basé sur le contenu du texte.
        Dans un modèle fully-trained, ce serait un vrai Tokenizer + Embedding.
        Ici, nous utilisons une fonction de hachage déterministe pour remplir
        le segment 'Entities' [0:64] du Mentalese, respectant l'Orthogonal Sparsity.
        """
        qpls_vector = torch.zeros(1, 1, self.d_model)
        
        # Hachage très simple : on utilise la valeur ASCII des mots pour créer un vecteur dense
        words = text.split()[:64] # Limité à 64 mots pour le segment entité
        for i, word in enumerate(words):
            hash_val = sum(ord(c) for c in word) % 100 / 100.0 # Val entre 0 et 1
            qpls_vector[0, 0, i] = hash_val
            
        # Metadata: Source = Web (Dimension 192 = 3.0 pour Web)
        qpls_vector[0, 0, 192] = 3.0
        
        return qpls_vector
