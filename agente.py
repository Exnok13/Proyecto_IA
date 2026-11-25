from arbol_decision import ArbolDecisionID3


class AgenteBuscaminas:
    """
    Agente inteligente que utiliza un Árbol de Decisión ID3 entrenado para
    seleccionar la casilla con menor riesgo de mina en cada turno.
    """
    def __init__(self, modelo_id3: ArbolDecisionID3):
        self.modelo = modelo_id3

    def _contar_vecinos_descubiertos(self, ventana):
        """
        Cuenta cuántas casillas dentro de la ventana 5x5 ya están reveladas (valores 0 a 8).
        Se usa como criterio de desempate cuando varias casillas tienen igual riesgo.
        """
        return sum(1 for val in ventana if val >= 0)

    def elegir_movimiento(self, candidatos_frontera):
        """
        Evalúa todas las casillas ocultas de la frontera y elige la más segura.

        Parámetros:
            candidatos_frontera: lista de tuplas (fila, col, ventana_5x5), donde
                                 ventana_5x5 es una lista de 24 enteros.
        Retorna:
            Tupla (fila, col) con las coordenadas del mejor clic.
        """
        if not candidatos_frontera:
            raise ValueError("La lista de candidatos de la frontera está vacía.")

        mejor_coord = None
        menor_riesgo = float("inf")
        max_contexto = -1

        for fila, col, ventana in candidatos_frontera:
            # Obtener probabilidad de que haya una mina en (fila, col)
            prob_mina = self.modelo.predecir_probabilidad_uno(ventana)
            vecinos_abiertos = self._contar_vecinos_descubiertos(ventana)

            # Seleccionar la casilla con menor probabilidad de mina.
            # En caso de empate exacto en probabilidad, preferir la casilla que tenga
            # más vecinos descubiertos (mayor información confiable alrededor).
            if (prob_mina < menor_riesgo) or (
                prob_mina == menor_riesgo and vecinos_abiertos > max_contexto
            ):
                menor_riesgo = prob_mina
                max_contexto = vecinos_abiertos
                mejor_coord = (fila, col)

        return mejor_coord


# --- Ejemplo rápido de prueba unitaria (Play Tennis) ---
if __name__ == "__main__":
    # Prueba rápida para verificar que ID3 calcula bien la entropía y construye el árbol
    # Atributos codificados: Outlook (0:Sunny, 1:Overcast, 2:Rain), Humidity (0:High, 1:Normal), Wind (0:Weak, 1:Strong)
    X_ejemplo = [
        [0, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0], [2, 1, 0],
        [2, 1, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0], [2, 1, 0],
        [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 0, 1]
    ]
    y_ejemplo = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

    arbol = ArbolDecisionID3(max_profundidad=3, min_muestras_division=2)
    arbol.fit(X_ejemplo, y_ejemplo)

    print("Entropía inicial:", round(arbol._calcular_entropia(y_ejemplo), 4))  # Debe dar ~0.9403
    print("Atributo raíz elegido (0=Outlook):", arbol.raiz.indice_caracteristica)  # Debe dar 0
    print("Predicciones sobre entrenamiento:", arbol.predecir(X_ejemplo))