import math
import pickle
import random
from collections import defaultdict


class Nodo:
    """
    Representa un nodo dentro del Árbol de Decisión ID3.
    """
    def __init__(self, indice_caracteristica=None, prob_mina=0.0, es_hoja=False):
        self.indice_caracteristica = indice_caracteristica
        self.hijos = {}
        self.prob_mina = prob_mina
        self.es_hoja = es_hoja


class ArbolDecisionID3:
    """
    Implementación desde cero del algoritmo ID3 para clasificación binaria
    (0 = casilla segura, 1 = mina) con salida probabilística.
    """
    def __init__(self, max_profundidad=8, min_muestras_division=10, max_caracteristicas=None):
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        # Si es None, evalúa todas las características (ID3 estándar).
        # Si es un entero, evalúa un subconjunto aleatorio en cada nodo (para Random Forest).
        self.max_caracteristicas = max_caracteristicas
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

        grupos = defaultdict(list)
        for fila, etiqueta in zip(X, y):
            grupos[fila[indice_caracteristica]].append(etiqueta)

        entropia_hijos = 0.0
        for etiquetas_subgrupo in grupos.values():
            peso = len(etiquetas_subgrupo) / total
            entropia_hijos += peso * self._calcular_entropia(etiquetas_subgrupo)

        return entropia_padre - entropia_hijos

    def _mejor_caracteristica(self, X, y, caracteristicas_disponibles):
        """
        Evalúa las características candidatas y retorna el índice de la que
        aporta la mayor Ganancia de Información.
        """
        candidatas = caracteristicas_disponibles
        if (
            self.max_caracteristicas is not None
            and len(caracteristicas_disponibles) > self.max_caracteristicas
        ):
            candidatas = random.sample(caracteristicas_disponibles, self.max_caracteristicas)

        mejor_ganancia = -1.0
        mejor_indice = None

        for idx in candidatas:
            ganancia = self._ganancia_informacion(X, y, idx)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_indice = idx

        return mejor_indice, mejor_ganancia

    def fit(self, X, y):
        """
        Entrena el árbol de decisión a partir de X e y.
        """
        if not X or not y:
            raise ValueError("Los datos de entrenamiento X e y no pueden estar vacíos.")

        num_caracteristicas = len(X[0])
        caracteristicas_disponibles = list(range(num_caracteristicas))
        self.raiz = self._construir_arbol(X, y, caracteristicas_disponibles, profundidad=0)

    def _construir_arbol(self, X, y, caracteristicas_disponibles, profundidad):
        """
        Método recursivo que construye los nodos aplicando ID3 y criterios de parada.
        """
        total = len(y)
        prob_mina = sum(y) / total

        if (
            prob_mina == 0.0
            or prob_mina == 1.0
            or profundidad >= self.max_profundidad
            or total < self.min_muestras_division
            or not caracteristicas_disponibles
        ):
            return Nodo(prob_mina=prob_mina, es_hoja=True)

        mejor_idx, mejor_ganancia = self._mejor_caracteristica(X, y, caracteristicas_disponibles)

        if mejor_idx is None or mejor_ganancia <= 0.0:
            return Nodo(prob_mina=prob_mina, es_hoja=True)

        nodo = Nodo(indice_caracteristica=mejor_idx, prob_mina=prob_mina, es_hoja=False)
        nuevas_disponibles = [c for c in caracteristicas_disponibles if c != mejor_idx]

        particiones_X = defaultdict(list)
        particiones_y = defaultdict(list)
        for fila, etiqueta in zip(X, y):
            val = fila[mejor_idx]
            particiones_X[val].append(fila)
            particiones_y[val].append(etiqueta)

        for val in particiones_X:
            nodo.hijos[val] = self._construir_arbol(
                particiones_X[val],
                particiones_y[val],
                nuevas_disponibles,
                profundidad + 1,
            )

        return nodo

    def predecir_probabilidad_uno(self, x):
        """
        Retorna la probabilidad estimada de mina (0.0 a 1.0) para una muestra 'x'.
        """
        if self.raiz is None:
            raise RuntimeError("El árbol debe ser entrenado con fit() antes de predecir.")

        nodo_actual = self.raiz
        while not nodo_actual.es_hoja:
            valor_celda = x[nodo_actual.indice_caracteristica]
            if valor_celda in nodo_actual.hijos:
                nodo_actual = nodo_actual.hijos[valor_celda]
            else:
                break

        return nodo_actual.prob_mina

    def predecir_probabilidades(self, X):
        return [self.predecir_probabilidad_uno(fila) for fila in X]

    def predecir(self, X, umbral=0.5):
        return [1 if prob >= umbral else 0 for prob in self.predecir_probabilidades(X)]

    def guardar_modelo(self, ruta_archivo):
        with open(ruta_archivo, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def cargar_modelo(ruta_archivo):
        with open(ruta_archivo, "rb") as f:
            return pickle.load(f)


class BosqueAleatorioID3:
    """
    Ensamble (Random Forest) compuesto por múltiples Árboles de Decisión ID3
    entrenados sobre muestras Bootstrap y subconjuntos aleatorios de atributos.
    """
    def __init__(self, num_arboles=7, max_profundidad=9, min_muestras_division=15, max_caracteristicas=12, semilla=42):
        self.num_arboles = num_arboles
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.max_caracteristicas = max_caracteristicas
        self.semilla = semilla
        self.arboles = []

    def fit(self, X, y):
        """
        Entrena cada árbol ID3 del bosque con una muestra aleatoria con reemplazo.
        """
        random.seed(self.semilla)
        self.arboles = []
        n = len(X)

        for _ in range(self.num_arboles):
            # Muestreo Bootstrap (con reemplazo) del 85% del dataset
            tam_muestra = int(n * 0.85)
            indices = [random.randint(0, n - 1) for _ in range(tam_muestra)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            arbol = ArbolDecisionID3(
                max_profundidad=self.max_profundidad,
                min_muestras_division=self.min_muestras_division,
                max_caracteristicas=self.max_caracteristicas,
            )
            arbol.fit(X_boot, y_boot)
            self.arboles.append(arbol)

    def predecir_probabilidad_uno(self, x):
        """
        Promedia la probabilidad de mina predicha por todos los árboles del bosque.
        """
        suma_probs = sum(arbol.predecir_probabilidad_uno(x) for arbol in self.arboles)
        return suma_probs / len(self.arboles)

    def predecir_probabilidades(self, X):
        return [self.predecir_probabilidad_uno(fila) for fila in X]

    def predecir(self, X, umbral=0.5):
        return [1 if prob >= umbral else 0 for prob in self.predecir_probabilidades(X)]

    def guardar_modelo(self, ruta_archivo):
        with open(ruta_archivo, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def cargar_modelo(ruta_archivo):
        with open(ruta_archivo, "rb") as f:
            return pickle.load(f)