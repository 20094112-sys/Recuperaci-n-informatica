# Mi primer modelo de Machine Learning

# 1. Importar las herramientas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2. Cargar los datos de flores
flores = load_iris()
X = flores.data
y = flores.target

# 3. Separar los datos para entrenar y probar
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Crear el modelo
modelo = DecisionTreeClassifier(random_state=42)

# 5. Entrenar el modelo
modelo.fit(X_entrenamiento, y_entrenamiento)

# 6. Realizar predicciones
predicciones = modelo.predict(X_prueba)

# 7. Evaluar el resultado
precision = accuracy_score(y_prueba, predicciones)

print("Modelo entrenado correctamente")
print("Precisión del modelo:", precision * 100, "%")

# 8. Probar una nueva flor
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]
resultado = modelo.predict(nueva_flor)

print("La flor pertenece a:", flores.target_names[resultado[0]])
