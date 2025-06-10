import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Dados base
dados = {
    "Nome": ["João Silva", "Ana Souza", "Pedro Lima", "Lucas Rocha", "Carla Mendes",
             "Bruno Costa", "Mariana Dias", "Rafael Torres", "Beatriz Alves",
             "Felipe Martins", "Gabriela Santos", "Henrique Lima", "Isabela Costa",
             "João Pedro", "Larissa Souza"],
    "Idade": [14, 13, 15, 12, 14, 13, 12, 14, 13, 15, 14, 13, 15, 12, 14],
    "Altura": [165, 158, 172, 150, 160, 163, 155, 168, 162, 170, 159, 164, 167, 153, 161],
    "Peso": [55, 52, 65, 48, 56, 60, 54, 70, 59, 68, 54, 63, 66, 50, 58],
    "%Gordura": [14, 18, 16, 20, 17, 15, 22, 19, 16, 17, 15, 18, 17, 21, 16],
    "Condicionamento": [4, 3, 5, 2, 4, 5, 3, 2, 4, 5, 4, 3, 5, 2, 4],
    "IMC": [20.2, 20.8, 22.0, 21.3, 21.9, 22.6, 22.5, 24.8, 22.5, 23.5, 21.4, 23.4, 23.7, 21.4, 22.4],
    "Classificação": ["Apto", "Não Apto", "Apto", "Não Apto", "Apto", "Apto", "Não Apto",
                      "Não Apto", "Apto", "Apto", "Apto", "Não Apto", "Apto", "Não Apto", "Apto"]
}

df = pd.DataFrame(dados)

# Features usadas
X = df[["Idade", "Altura", "Peso", "%Gordura", "Condicionamento", "IMC"]]
y = df["Classificação"]

# Novo exemplo: Rafael Santos
rafael = np.array([[14, 170, 68, 16, 5, 23.5]])

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rafael_scaled = scaler.transform(rafael)

# Treinar KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Previsão
predicao = knn.predict(rafael_scaled)
print(f"Rafael Santos foi classificado como: {predicao[0]}")

# Pegar os 2 componentes principais para visualização
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
rafael_pca = pca.transform(rafael_scaled)

# Adicionar ao DataFrame para plotagem
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Classificação"] = y
df_plot["Nome"] = df["Nome"]

# Gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Classificação", style="Classificação", s=100)

# Rafael
plt.scatter(rafael_pca[0, 0], rafael_pca[0, 1], color="black", s=200, label="Rafael Santos", marker="X")
plt.title("Classificação KNN com K=3")
plt.legend()
plt.grid(True)
plt.show()
