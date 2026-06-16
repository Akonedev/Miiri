import json
import os
import subprocess
import urllib.request

def main():
    mcp_path = os.path.expanduser("~/.gemini/mcp.json")
    print(f"[*] Lecture du fichier {mcp_path}...")
    
    if not os.path.exists(mcp_path):
        print(f"[ERREUR] Le fichier {mcp_path} n'existe pas.")
        return

    try:
        with open(mcp_path, 'r') as f:
            mcp_data = json.load(f)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le fichier JSON: {e}")
        return

    token = None
    try:
        # Recherche du token dans la config github
        env_vars = mcp_data.get("mcpServers", {}).get("github", {}).get("env", {})
        token = env_vars.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not token:
            print("[ERREUR] Token GITHUB_PERSONAL_ACCESS_TOKEN introuvable dans le serveur 'github'.")
            return
    except Exception as e:
        print(f"[ERREUR] Structure JSON inattendue: {e}")
        return

    print("[*] Token trouvé. Authentification auprès de GitHub API...")
    
    req = urllib.request.Request("https://api.github.com/user")
    req.add_header("Authorization", f"token {token}")
    
    try:
        with urllib.request.urlopen(req) as response:
            user_data = json.loads(response.read().decode('utf-8'))
            username = user_data.get("login")
            print(f"[*] Authentifié en tant que : {username}")
    except Exception as e:
        print(f"[ERREUR] L'authentification a échoué. Le token est-il valide ? {e}")
        return

    print("[*] Création du dépôt public Miiri-256 sur GitHub...")
    repo_req = urllib.request.Request("https://api.github.com/user/repos", method="POST")
    repo_req.add_header("Authorization", f"token {token}")
    repo_req.add_header("Content-Type", "application/json")
    payload = json.dumps({
        "name": "Miiri-256",
        "description": "Architecture Miiri (Pensée Unifiée) (Miiri-256) - Native Unified AGI Architecture",
        "private": False
    }).encode('utf-8')
    
    try:
        with urllib.request.urlopen(repo_req, data=payload) as response:
            print("[*] Dépôt créé avec succès.")
    except urllib.error.HTTPError as e:
        if e.code == 422:
            print("[*] Le dépôt existe déjà sur GitHub. Continuation...")
        else:
            print(f"[ERREUR] Échec de la création du dépôt: {e.code} - {e.read().decode('utf-8')}")
            return
            
    print("[*] Configuration du remote et push du code...")
    push_url = f"https://{username}:{token}@github.com/{username}/Miiri-256.git"
    clean_url = f"https://github.com/{username}/Miiri-256.git"
    
    try:
        subprocess.run(["git", "remote", "remove", "origin"], stderr=subprocess.DEVNULL)
    except:
        pass
        
    try:
        subprocess.run(["git", "remote", "add", "origin", push_url], check=True)
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        print("[*] Uploading en cours (Pushing to main)...")
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        # Clean remote url
        subprocess.run(["git", "remote", "set-url", "origin", clean_url], check=True)
        print("\n[SUCCESS] ✨ Tout le code de l'architecture Miiri-256 a été poussé sur votre GitHub avec succès !")
        print(f"URL de votre dépôt : {clean_url}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERREUR] Le git push a échoué. ({e})")
        # Ensure we clean the URL even on failure
        subprocess.run(["git", "remote", "set-url", "origin", clean_url], stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
