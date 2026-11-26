def enriquecer_ventana(ventana_24):
    """
    Toma los 24 valores crudos de la ventana 5x5 y calcula 6 características
    resumidas discretas sobre el vecindario inmediato 3x3 (8 vecinos directos)
    para facilitar el aprendizaje del árbol ID3.

    En el vector de 24 elementos (ordenado fila por fila excluyendo el centro):
    Los índices de los 8 vecinos directos (anillo 3x3) son:
    [6, 7, 8, 11, 12, 15, 16, 17]
    """
    indices_3x3 = (6, 7, 8, 11, 12, 15, 16, 17)
    vecinos_directos = [ventana_24[i] for i in indices_3x3]

    ocultos = sum(1 for v in vecinos_directos if v == -1)
    descubiertos = [v for v in vecinos_directos if v >= 0]
    num_descubiertos = len(descubiertos)
    suma_numeros = sum(descubiertos)
    max_numero = max(descubiertos) if descubiertos else -1
    tiene_cero = 1 if 0 in descubiertos else 0

    # Categoría discreta de presión de minas alrededor (0: baja, 1: media, 2: alta)
    # Relaciona la suma de números revelados con los espacios ocultos disponibles
    if ocultos == 0 or suma_numeros == 0:
        nivel_presion = 0
    else:
        ratio = suma_numeros / ocultos
        if ratio < 0.5:
            nivel_presion = 1
        elif ratio < 1.0:
            nivel_presion = 2
        else:
            nivel_presion = 3

    caracteristicas_extra = [
        ocultos,
        num_descubiertos,
        suma_numeros,
        max_numero,
        tiene_cero,
        nivel_presion,
    ]
    return ventana_24 + caracteristicas_extra


class AgenteBuscaminas:
    """
    Agente Híbrido para Buscaminas:
    1. Aplica deducción lógica determinista para identificar minas y casillas 100% seguras.
    2. Cuando no hay jugada determinista, usa el modelo ID3 / Bosque Aleatorio
       para elegir la casilla con menor probabilidad de mina.
    """
    def __init__(self, modelo_id3):
        self.modelo = modelo_id3
        self.banderas = set()

    def reiniciar_partida(self):
        """Limpia las banderas recordadas al comenzar un nuevo tablero."""
        self.banderas = set()

    def _deducir_movimiento_seguro(self, juego):
        """
        Revisa los números visibles en el tablero:
        - Si un número N tiene exactamente N vecinos ocultos, todos son minas (banderas).
        - Si un número N ya tiene N banderas alrededor, sus demás vecinos ocultos son 100% seguros.
        Retorna una tupla (fila, col) segura si encuentra una, o None si debe usar ID3.
        """
        cambio = True
        while cambio:
            cambio = False
            for f in range(juego.filas):
                for c in range(juego.columnas):
                    if not juego.visible[f][c]:
                        continue
                    valor = juego.tablero[f][c]
                    if valor <= 0:
                        continue

                    vecinos = juego._get_vecinos(f, c)
                    ocultos_sin_bandera = [
                        (nf, nc)
                        for nf, nc in vecinos
                        if not juego.visible[nf][nc] and (nf, nc) not in self.banderas
                    ]
                    banderas_adyacentes = sum(
                        1 for nf, nc in vecinos if (nf, nc) in self.banderas
                    )

                    # Regla 1: Las casillas ocultas restantes coinciden con las minas faltantes
                    minas_faltantes = valor - banderas_adyacentes
                    if minas_faltantes > 0 and len(ocultos_sin_bandera) == minas_faltantes:
                        for coord in ocultos_sin_bandera:
                            if coord not in self.banderas:
                                self.banderas.add(coord)
                                cambio = True

                    # Regla 2: Todas las minas de este número ya están marcadas -> abrir el resto
                    elif minas_faltantes == 0 and len(ocultos_sin_bandera) > 0:
                        return ocultos_sin_bandera[0]

        return None

    def _contar_vecinos_descubiertos(self, ventana_24):
        return sum(1 for val in ventana_24 if val >= 0)

    def elegir_movimiento(self, candidatos_frontera, juego=None):
        """
        Elige la mejor jugada combinando deducción lógica (si se entrega 'juego')
        y predicción probabilística con el modelo ID3.
        """
        if not candidatos_frontera:
            raise ValueError("La lista de candidatos de la frontera está vacía.")

        # Paso 1: Intentar deducción lógica 100% segura
        if juego is not None:
            jugada_segura = self._deducir_movimiento_seguro(juego)
            if jugada_segura is not None:
                return jugada_segura

        # Paso 2: Evaluar con el modelo ID3 (ignorando casillas ya marcadas como bandera)
        candidatos_validos = [
            (f, c, v) for f, c, v in candidatos_frontera if (f, c) not in self.banderas
        ]
        if not candidatos_validos:
            candidatos_validos = candidatos_frontera

        mejor_coord = None
        menor_riesgo = float("inf")
        max_contexto = -1

        for fila, col, ventana_24 in candidatos_validos:
            ventana_enriquecida = enriquecer_ventana(ventana_24)
            prob_mina = self.modelo.predecir_probabilidad_uno(ventana_enriquecida)
            vecinos_abiertos = self._contar_vecinos_descubiertos(ventana_24)

            if (prob_mina < menor_riesgo) or (
                prob_mina == menor_riesgo and vecinos_abiertos > max_contexto
            ):
                menor_riesgo = prob_mina
                max_contexto = vecinos_abiertos
                mejor_coord = (fila, col)

        return mejor_coord