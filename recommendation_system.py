
# Importar las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Cargar los datos (reemplaza dataset de productos)
# Este código está diseñado para generar productos que pueden ser generan
data = pd.read_csv('products.csv')

# Procesamiento de datos
# Seleccionar características y la etiqueta
features = data[['feature1', 'feature2', 'feature3']]
labels = data['label']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear el modelo
model = NearestNeighbors(n_neighbors=5)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Función para obtener recomendaciones
def get_recommendations(product_features):
    prediction = model.predict([product_features])
    return prediction

# Ejemplo de uso
example_product = [1.5, 2.3, 3.8]
recommended_product = get_recommendations(example_product)
print(f'Recommended Product: {recommended_product}')
