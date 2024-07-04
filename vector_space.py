import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

def attention_score(query, key):
    return np.dot(query, key) / (np.linalg.norm(query) * np.linalg.norm(key))

def gravity_force(m1, m2, r):
    G = 6.67430e-11 
    return G * m1 * m2 / (r**2)

sentence = "Attention Is All We Need"
words = sentence.split()
vectorizer = CountVectorizer().fit(words)
word_vectors = vectorizer.transform(words).toarray()
pca = PCA(n_components=3)
word_coords = pca.fit_transform(word_vectors)
word_coords = word_coords / np.abs(word_coords).max()
need_index = words.index("Need")
initial_embedding = word_coords[need_index]

## FIGURESSSSS

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Embedding Vector Analysis for 'Need'")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)

for word, coord in zip(words, word_coords):
    ax.scatter(coord[0], coord[1], coord[2], c='b', alpha=0.5)
    ax.text(coord[0], coord[1], coord[2], word, fontsize=8)
    
query = word_coords[need_index]
attention_weights = []

for coord in word_coords:
    score = attention_score(query, coord)
    attention_weights.append(score)

attention_weights = np.array(attention_weights)
attention_weights = attention_weights / attention_weights.sum() 
contextual_embedding_attention = np.sum(word_coords * attention_weights[:, np.newaxis], axis=0)


word_masses = np.array([len(word) * 10 for word in words]) 
need_mass = word_masses[need_index]
gravity_weights = []

for coord, mass in zip(word_coords, word_masses):
    r = np.linalg.norm(coord - word_coords[need_index])
    if r == 0:
        force = 0
    else:
        force = gravity_force(need_mass, mass, r)
    gravity_weights.append(force)

gravity_weights = np.array(gravity_weights)
gravity_weights = gravity_weights / gravity_weights.sum()  
contextual_embedding_gravity = np.sum(word_coords * gravity_weights[:, np.newaxis], axis=0)


ax.quiver(0, 0, 0, initial_embedding[0], initial_embedding[1], initial_embedding[2], 
          color='r', arrow_length_ratio=0.1, label='Initial Embedding')
ax.quiver(0, 0, 0, contextual_embedding_attention[0], contextual_embedding_attention[1], contextual_embedding_attention[2], 
          color='g', arrow_length_ratio=0.1, label='Self-Attention Embedding')
ax.quiver(0, 0, 0, contextual_embedding_gravity[0], contextual_embedding_gravity[1], contextual_embedding_gravity[2], 
          color='m', arrow_length_ratio=0.1, label='Gravity-based Embedding')

ax.legend()

plt.tight_layout()
plt.show()

print("Initial embedding for 'need':", initial_embedding)
print("Attention-based contextual embedding for 'need':", contextual_embedding_attention)
print("Gravity-based contextual embedding for 'need':", contextual_embedding_gravity)