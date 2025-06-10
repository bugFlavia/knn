import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Dataset
data = {
    "Idade": [20, 22, 25, 30, 35],
    "Tempo": [5.0, 5.2, 5.4, 6.0, 6.2]
}

df = pd.DataFrame(data)

# Separar variáveis
X = df[["Idade"]]  # Feature
y = df["Tempo"]    # Target

# Novo dado para prever
idade_nova = np.array([[24]])

# Treinar modelo KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X, y)

# Prever tempo para idade 24
tempo_previsto = knn_reg.predict(idade_nova)
print(f"Tempo previsto para idade 24 anos: {tempo_previsto[0]:.2f} minutos")

# Visualização
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Dados de treino")
plt.scatter(idade_nova, tempo_previsto, color="red", label="Previsão (Idade 24)", marker='X', s=100)
plt.plot(X, knn_reg.predict(X), color="green", linestyle='--', label="KNN (k=3)")
plt.title("KNN Regressão: Previsão de Tempo com Base na Idade")
plt.xlabel("Idade")
plt.ylabel("Tempo (min)")
plt.legend()
plt.grid(True)
plt.show()
