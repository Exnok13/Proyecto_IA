import math
import pickle
from collections import defaultdict


class Nodo:
    """
    Representa un nodo dentro del Árbol de Decisión ID3.
    """
    def __init__(self, indice_caracteristica=None, prob_mina=0.0, es_hoja=False):
        # Índice de la celda en el vector de entrada (0 a 23 para ventana 5x5)
        self.indice_caracteristica = indice_caracteristica
        # Diccionario que mapea {valor_discreto_celda: Nodo_hijo}
        self.hijos = {}
        # Probabilidad de que la casilla central sea mina en este nodo (0.0 a 1.0)
        self.prob_mina = prob_mina
        # Indica si este nodo es terminal (no se divide más)
        self.es_hoja = es_hoja


class ArbolDecisionID3:
    """
    Implementación desde cero del algoritmo ID3 para clasificación binaria
    (0 = casilla segura, 1 = mina) con salida probabilística.
    """
    def __init__(self, max_profundidad=8, min_muestras_division=10):
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.raiz = None

    def _calcular_entropia(self, y):
        """
        Calcula la entropía de Shannon H(S) para una lista de etiquetas binarias (0 o 1).
        """
        total = len(y)
        if total == 0:
            return 0.0

        minas = sum(y)
        seguras = total - minas

        # Si el conjunto es puro (solo minas o solo seguras), la entropía es 0
        if minas == 0 or seguras == 0:
            return 0.0

        p_mina = minas / total
        p_segura = seguras / total

        return -(p_mina * math.log2(p_mina) + p_segura * math.log2(p_segura))

    def _ganancia_informacion(self, X, y, indice_caracteristica):
        """
        Calcula la Ganancia de Información (Information Gain) al dividir los datos
        según los valores discretos de una característica específica.
        """
        entropia_padre = self._calcular_entropia(y)
        total = len(y)

        # Agrupar las etiquetas 'y' según el valor que toma la característica en cada fila
        grupos = defaultdict(list)
        for fila, etiqueta in zip(X, y):
            valor_celda = fila[indice_caracteristica]
            grupos[valor_celda].append(etiqueta)

        # Calcular la entropía ponderada de todos los subgrupos resultantes
        entropia_hijos = 0.0
        for etiquetas_subgrupo in grupos.values():
            peso = len(etiquetas_subgrupo) / total
            entropia_hijos += peso * self._calcular_entropia(etiquetas_subgrupo)

        return entropia_padre - entropia_hijos

    def _mejor_caracteristica(self, X, y, caracteristicas_disponibles):
        """
        Evalúa todas las características disponibles y retorna el índice de la que
        aporta la mayor Ganancia de Información junto con el valor de dicha ganancia.
        """
        mejor_ganancia = -1.0
        mejor_indice = None

        for idx in caracteristicas_disponibles:
            ganancia = self._ganancia_informacion(X, y, idx)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_indice = idx

        return mejor_indice, mejor_ganancia

    def fit(self, X, y):
        """
        Entrena el árbol de decisión a partir de la matriz de características X
        y el vector de etiquetas y.
        """
        if not X or not y:
            raise ValueError("Los datos de entrenamiento X e y no pueden estar vacíos.")

        num_caracteristicas = len(X[0])
        caracteristicas_disponibles = list(range(num_caracteristicas))
        self.raiz = self._construir_arbol(X, y, caracteristicas_disponibles, profundidad=0)

    def _construir_arbol(self, X, y, caracteristicas_disponibles, profundidad):
        """
        Método recursivo que construye los nodos del árbol aplicando ID3 y
        criterios de parada para evitar sobreajuste (overfitting).
        """
        total = len(y)
        prob_mina = sum(y) / total

        # Criterios de parada (Casos Base):
        # 1. Nodo puro (todas son minas o todas son seguras)
        # 2. Se alcanzó la profundidad máxima permitida
        # 3. Hay muy pocas muestras para justificar una nueva división
        # 4. No quedan características disponibles por evaluar
        if (
            prob_mina == 0.0
            or prob_mina == 1.0
            or profundidad >= self.max_profundidad
            or total < self.min_muestras_division
            or not caracteristicas_disponibles
        ):
            return Nodo(prob_mina=prob_mina, es_hoja=True)

        # Buscar la mejor característica para dividir este nodo
        mejor_idx, mejor_ganancia = self._mejor_caracteristica(X, y, caracteristicas_disponibles)

        # Si ninguna característica reduce la entropía, convertir en hoja
        if mejor_idx is None or mejor_ganancia <= 0.0:
            return Nodo(prob_mina=prob_mina, es_hoja=True)

        nodo = Nodo(indice_caracteristica=mejor_idx, prob_mina=prob_mina, es_hoja=False)

        # En ID3 clásico, el atributo elegido no se vuelve a usar en la misma rama
        nuevas_disponibles = [c for c in caracteristicas_disponibles if c != mejor_idx]

        # Particionar X e y según cada valor discreto encontrado en la mejor característica
        particiones_X = defaultdict(list)
        particiones_y = defaultdict(list)
        for fila, etiqueta in zip(X, y):
            val = fila[mejor_idx]
            particiones_X[val].append(fila)
            particiones_y[val].append(etiqueta)

        # Construir recursivamente cada subárbol hijo
        for val in particiones_X:
            nodo.hijos[val] = self._construir_arbol(
                particiones_X[val],
                particiones_y[val],
                nuevas_disponibles,
                profundidad + 1
            )

        return nodo

    def predecir_probabilidad_uno(self, x):
        """
        Recorre el árbol para una sola ventana 'x' y retorna la probabilidad
        estimada de que haya una mina (entre 0.0 y 1.0).
        """
        if self.raiz is None:
            raise RuntimeError("El árbol debe ser entrenado con fit() antes de predecir.")

        nodo_actual = self.raiz
        while not nodo_actual.es_hoja:
            valor_celda = x[nodo_actual.indice_caracteristica]
            # Si el valor existe en las ramas aprendidas, seguimos bajando
            if valor_celda in nodo_actual.hijos:
                nodo_actual = nodo_actual.hijos[valor_celda]
            else:
                # Si aparece un valor no visto en entrenamiento, retornamos la probabilidad
                # acumulada del nodo padre como respaldo seguro
                break

        return nodo_actual.prob_mina

    def predecir_probabilidades(self, X):
        """
        Retorna una lista con la probabilidad de mina para cada fila en X.
        """
        return [self.predecir_probabilidad_uno(fila) for fila in X]

    def predecir(self, X, umbral=0.5):
        """
        Retorna predicciones binarias (0 o 1) usando un umbral de decisión.
        Útil para calcular Accuracy y Matriz de Confusión en el conjunto de prueba.
        """
        return [1 if prob >= umbral else 0 for prob in self.predecir_probabilidades(X)]

    def guardar_modelo(self, ruta_archivo):
        """
        Guarda el árbol entrenado en disco para reutilizarlo sin reentrenar.
        """
        with open(ruta_archivo, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def cargar_modelo(ruta_archivo):
        """
        Carga un árbol previamente guardado desde disco.
        """
        with open(ruta_archivo, "rb") as f:
            return pickle.load(f)