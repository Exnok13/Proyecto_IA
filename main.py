import csv
import time
import random

from Busca_minas.minesweeper import minasweeper
from arbol_decision import ArbolDecisionID3, BosqueAleatorioID3
from agente import AgenteBuscaminas, enriquecer_ventana


def cargar_csv_enriquecido(ruta_archivo):
    """
    Lee data/train.csv o data/test.csv y aplica ingeniería de características
    (de 24 a 30 atributos) en tiempo de carga.
    """
    X, y = [], []
    with open(ruta_archivo, mode="r", encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector)  # Saltar encabezados
        for fila in lector:
            if not fila:
                continue
            ventana_24 = [int(valor) for valor in fila[:-1]]
            etiqueta = int(fila[-1])
            X.append(enriquecer_ventana(ventana_24))
            y.append(etiqueta)
    return X, y


def calcular_metricas(y_real, y_pred):
    """
    Calcula Accuracy, Precision, Recall y Matriz de Confusión en Python puro.
    """
    tp = tn = fp = fn = 0
    for real, pred in zip(y_real, y_pred):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 0:
            tn += 1
        elif real == 0 and pred == 1:
            fp += 1
        elif real == 1 and pred == 0:
            fn += 1

    total = len(y_real)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "matriz": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def simular_partidas(agente=None, usar_deduccion=False, num_partidas=1000, semilla=42):
    """
    Simula partidas completas 9x9 con 10 minas.
    """
    random.seed(semilla)
    victorias = 0
    total_casillas_abiertas = 0

    for _ in range(num_partidas):
        juego = minasweeper(filas=9, columnas=9, can_minas=10)
        if agente is not None:
            agente.reiniciar_partida()

        # Primer clic seguro en el centro (4, 4)
        while juego.tablero[4][4] == -1:
            juego.reset()

        perdiste, ganaste = juego.jugada(4, 4)

        while not perdiste:
            frontera = juego.busqueda_casilla()
            if not frontera:
                ocultas = [
                    (f, c)
                    for f in range(juego.filas)
                    for c in range(juego.columnas)
                    if not juego.visible[f][c]
                ]
                if not ocultas:
                    break
                fila_elegida, col_elegida = random.choice(ocultas)
            else:
                if agente is None:
                    fila_elegida, col_elegida = random.choice(frontera)
                else:
                    candidatos = [
                        (f, c, juego.ventana_5x5(f, c)) for f, c in frontera
                    ]
                    ref_juego = juego if usar_deduccion else None
                    fila_elegida, col_elegida = agente.elegir_movimiento(
                        candidatos, juego=ref_juego
                    )

            perdiste, ganaste = juego.jugada(fila_elegida, col_elegida)

        if ganaste:
            victorias += 1
        total_casillas_abiertas += juego.casilla_descubiertas

    win_rate = (victorias / num_partidas) * 100
    promedio_casillas = total_casillas_abiertas / num_partidas
    return win_rate, promedio_casillas


if __name__ == "__main__":
    print("=== 1. CARGANDO Y ENRIQUECIENDO DATASETS (30 CARACTERÍSTICAS) ===")
    X_train, y_train = cargar_csv_enriquecido("data/train.csv")
    X_test, y_test = cargar_csv_enriquecido("data/test.csv")
    print(f"Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras\n")

    print("=== 2. ENTRENANDO ÁRBOL ID3 INDIVIDUAL VS BOSQUE ALEATORIO ID3 ===")
    inicio = time.time()
    arbol_unico = ArbolDecisionID3(max_profundidad=9, min_muestras_division=15)
    arbol_unico.fit(X_train, y_train)
    t_arbol = time.time() - inicio
    acc_arbol = calcular_metricas(y_test, arbol_unico.predecir(X_test))["accuracy"] * 100
    print(f"Árbol ID3 Único (Prof. 9)       -> Test Acc: {acc_arbol:.2f}% | Tiempo: {t_arbol:.2f}s")

    inicio = time.time()
    bosque = BosqueAleatorioID3(
        num_arboles=9,
        max_profundidad=10,
        min_muestras_division=12,
        max_caracteristicas=14,
        semilla=42,
    )
    bosque.fit(X_train, y_train)
    t_bosque = time.time() - inicio
    metricas_bosque = calcular_metricas(y_test, bosque.predecir(X_test))
    acc_bosque = metricas_bosque["accuracy"] * 100
    print(f"Bosque Aleatorio ID3 (9 árboles)-> Test Acc: {acc_bosque:.2f}% | Tiempo: {t_bosque:.2f}s")

    mejor_modelo = bosque if acc_bosque >= acc_arbol else arbol_unico
    mejor_modelo.guardar_modelo("modelo_id3.pkl")

    print("\n=== 3. MÉTRICAS DEL MEJOR MODELO EN TEST SET ===")
    print(f"Accuracy : {metricas_bosque['accuracy'] * 100:.2f}%")
    print(f"Precision: {metricas_bosque['precision'] * 100:.2f}%")
    print(f"Recall   : {metricas_bosque['recall'] * 100:.2f}%")
    print(f"Matriz de Confusión: {metricas_bosque['matriz']}")

    print("\n=== 4. SIMULACIÓN DE 1.000 PARTIDAS COMPLETAS ===")
    wr_azar, cas_azar = simular_partidas(agente=None, num_partidas=1000)
    print(f"1. Agente Aleatorio (Baseline)        -> Win Rate: {wr_azar:5.2f}% | Casillas: {cas_azar:.1f}/71")

    agente_solo_ml = AgenteBuscaminas(mejor_modelo)
    wr_ml, cas_ml = simular_partidas(agente=agente_solo_ml, usar_deduccion=False, num_partidas=1000)
    print(f"2. Agente ID3 Mejorado (Solo ML)      -> Win Rate: {wr_ml:5.2f}% | Casillas: {cas_ml:.1f}/71")

    agente_hibrido = AgenteBuscaminas(mejor_modelo)
    wr_hib, cas_hib = simular_partidas(agente=agente_hibrido, usar_deduccion=True, num_partidas=1000)
    print(f"3. Agente Híbrido (Deducción + ID3)   -> Win Rate: {wr_hib:5.2f}% | Casillas: {cas_hib:.1f}/71")