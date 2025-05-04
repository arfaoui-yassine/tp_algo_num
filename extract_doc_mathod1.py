import math
import numpy as np
import random

def produit_scalaire(v1, v2):
    return np.dot(v1, v2)

def module(v):
    return np.linalg.norm(v)

def find(content, i, j):
    while i < n - 3 and content[i:i + len(j)] != j:
        i += 1
    if i >= n - 3:
        return n
    return i


# Extraction des documents
with open('c:/Users/waela/OneDrive/Desktop/TP/ProjetL_python/Documents.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Stockage des documents et des termes dans une liste et un set
n = len(content)
doc = []
terms = set()
index1 = 0
nb = 2
while index1 < n - 1:
    index2 = find(content, index1, str(nb))
    doc.append(content[index1:index2][len(str(nb - 1)) + 1:-1])
    for i in doc[-1].split():
        terms.add(i)
    index1 = index2
    nb += 1
n = len(doc)

# Construction du vecteur des termes
terms = list(terms)
terms.sort()
m = len(terms)

# Dictionnaire pour donner l'indice de chaque terme
index_terms = dict()
for i in range(m):
    index_terms[terms[i]] = i

# Construction du modèle de l'espace vectoriel matriciel initialisé à 0
D = np.zeros((n, m), dtype=int)  # Utilisation de NumPy pour initialiser une matrice de zéros

# Application du modèle de l'approche binaire
for i in range(n):
    for j in doc[i].split():
        if j in index_terms and index_terms[j] < m:
            D[i, index_terms[j]] = 1

# Génération d'une requête aléatoire à partir des termes
random_query = random.sample(terms, 3)  # Générer une requête avec 5 termes aléatoires
print("Requête aléatoire générée :", random_query)

# Construction du vecteur de la requête
vect_q = np.zeros(m, dtype=int)
for i in random_query:
    if i in index_terms and index_terms[i] < m:
        vect_q[index_terms[i]] = 1

# Construction du vecteur des scores pour chaque document
score_q = np.zeros(n)
for i in range(n):
    if module(vect_q) > 0 and module(D[i]) > 0:
        score_q[i] = produit_scalaire(vect_q, D[i]) / (module(vect_q) * module(D[i]))

# Placer les documents avec un score >= 0.8 et les trier en fonction de leurs scores
doc_priority = []
for i in range(n):
    if score_q[i] >= 0.2:
        doc_priority.append((doc[i], score_q[i]))
doc_priority = sorted(doc_priority, reverse=True, key=lambda x: x[1])

# Affichage des documents en ordre de pertinence avec traitement des cas d'indisponibilité
if len(doc_priority) == 0:
    print("Aucun document pertinent trouvé.")
else:
    print("\nDocuments pertinents triés par score :")
    for i in doc_priority[:min(6,len(doc_priority))]:
        print(f"- {i[0]} (score = {i[1]:.2f})")