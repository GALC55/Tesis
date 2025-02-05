import re
import emoji
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np


# =====================
# 1. Limpieza de Texto
# =====================
def clean_tweet(text):
    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Eliminar menciones (@usuario)
    text = re.sub(r"@\w+", "", text)
    # Eliminar emojis
    text = emoji.replace_emoji(text, replace="")
    # Eliminar caracteres especiales y números
    text = re.sub(r"[^A-Za-z\s#]", "", text)
    # Convertir a minúsculas
    text = text.lower()
    return text.strip()


# =====================
# 2. Cargar Dataset
# =====================
# Dataset de ejemplo
data = pd.read_excel('datos_es_unidos.xlsx')

df = data

## Combinar las columnas 'text' y 'hashtags' antes de limpiar
df["combined_text"] = df["text"] + " " + df["hashtags"].fillna("")
df["clean_text"] = df["combined_text"].apply(clean_tweet)

# Imprimir resultados
print("Texto combinado y limpio:")
print(df[["text", "hashtags", "combined_text", "clean_text"]])

# =====================
# 3. Tokenización y Embeddings con RoBERTa
# =====================
# Cargar el tokenizador y modelo RoBERTa
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)


def get_embeddings(texts, tokenizer, model):
    embeddings = []
    model.eval()  # Modo de evaluación
    with torch.no_grad():
        for text in texts:
            # Tokenizar el texto
            tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            # Obtener embeddings del [CLS] token
            outputs = model(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Embedding del token [CLS]
            embeddings.append(cls_embedding.squeeze().numpy())
    return embeddings


# Generar embeddings
clean_texts = df["clean_text"].tolist()
embeddings = get_embeddings(clean_texts, tokenizer, model)


# =====================
# 4. Clustering con 3 Clusters
# =====================
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Agregar etiquetas al DataFrame
df["cluster"] = cluster_labels

# =====================
# Imprimir Parámetros del K-Means
# =====================
print("\n=== Parámetros del K-Means ===")
print(f"Centroides de los clusters:\n{kmeans.cluster_centers_}")
print(f"Inercia (compactación de los clusters): {kmeans.inertia_}")
print(f"Etiquetas asignadas a los datos:\n{cluster_labels}")
print(f"Iteraciones hasta convergencia: {kmeans.n_iter_}")

df.to_excel('datos_con_cluster.xlsx')
# =====================
# Evaluación de Clustering
# =====================
# Silhouette Score (cohesión entre clusters)
silhouette_avg = silhouette_score(embeddings, cluster_labels)
print(f"\nSilhouette Score: {silhouette_avg}")

# Calinski-Harabasz Index (diferenciación entre clusters)
calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
print(f"Calinski-Harabasz Index: {calinski_harabasz}")

# =====================
# Visualización del Count de Clusters en Gráfico de Barras
# =====================
# Contar la cantidad de tweets en cada cluster
cluster_counts = df['cluster'].value_counts()

# Crear un gráfico de barras
plt.figure(figsize=(8, 6))
cluster_counts.plot(kind='bar', color='skyblue')
plt.title('Conteo de Tweets en Cada Cluster')
plt.xlabel('Cluster')
plt.ylabel('Número de Tweets')
plt.xticks(rotation=0)
plt.show()

# =====================
# Visualización
# =====================
# Reducir dimensionalidad para visualización (PCA)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Crear un gráfico de dispersión
plt.figure(figsize=(8, 6))
for i in range(3):
    cluster_points = reduced_embeddings[df["cluster"] == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

plt.title("Clustering de Tweets (RoBERTa + K-Means) - 3 Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()