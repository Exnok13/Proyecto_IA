import csv
import os
import random
import sys
import matplotlib.pyplot as plt
"""
Módulo de Evaluación y Visualización
Evalúa el desempeño del Accuracy, Precision, Recall, F1-Score, Matriz de Confusión
y dinámico el Win Rate en 1.000 partidas simuladas comparando el Árbol de Decisión ID3
individual frente al Agente Híbrido.
"""
# Permite importar agente.py y arbol_decision/ desde la carpeta raíz del proyecto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Busca_minas.minesweeper import minasweeper
from arbol_decision import ArbolDecisionID3, BosqueAleatorioID3
from agente import AgenteBuscaminas, enriquecer_ventana


def cargar_csv(ruta_archivo):
    """Carga los datasets de entrenamiento o prueba y aplica enriquecimiento de 24 a 30 características."""
    X, y = [], []
    with open(ruta_archivo, mode="r", encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector)
        for fila in lector:
            if not fila:
                continue
            ventana_24 = [int(v) for v in fila[:-1]]
            X.append(enriquecer_ventana(ventana_24))
            y.append(int(fila[-1]))
    return X, y


def calcular_metricas(y_true: list, y_pred: list) -> dict:
    """Calcula Accuracy, Precision, Recall, F1-Score y Matriz de Confusión."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def simulador_jugadas(agente, usar_deduccion=False, num_partidas=1000, semilla=42):
    """Simula partidas completas de Buscaminas y devuelve estadísticas de desempeño."""
    random.seed(semilla)
    victorias = 0
    total_casillas_abiertas = 0

    for _ in range(num_partidas):
        juego = minasweeper(filas=9, columnas=9, can_minas=10)
        agente.reiniciar_partida()

        perdiste, ganaste = juego.jugada(random.randint(0, 8), random.randint(0, 8))

        while not perdiste:
            # Obtiene casillas ocultas adyacentes a números revelados
            frontera = juego.busqueda_casilla()
            if not frontera:
                # Si no hay pistas adyacentes, elige al azar entre ocultas
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
                # Extrae las ventanas 5x5 de la frontera y consulta al agente
                candidatos = [(f, c, juego.ventana_5x5(f, c)) for f, c in frontera]
                ref_juego = juego if usar_deduccion else None
                fila_elegida, col_elegida = agente.elegir_movimiento(candidatos, juego=ref_juego)

            perdiste, ganaste = juego.jugada(fila_elegida, col_elegida)

        if ganaste:
            victorias += 1
        total_casillas_abiertas += juego.casilla_descubiertas

    return {
        "win_rate": (victorias / num_partidas) * 100,
        "avg_cells": total_casillas_abiertas / num_partidas,
    }


def generar_graficas(depths,train_accs,test_accs,hybrid_model_metrics,tree_stats,hibrido_stats,out_dir="report",):
    """Genera y guarda los 3 gráficos comparativos entre el Árbol ID3 y el Agente Híbrido."""
    os.makedirs(out_dir, exist_ok=True)

    # Grafico de Overfitting 
    plt.figure(figsize=(8, 5))
    plt.plot(depths, [a * 100 for a in train_accs], marker="o", label="Training Accuracy", linewidth=2)
    plt.plot(depths, [a * 100 for a in test_accs], marker="s", label="Test Accuracy", linewidth=2)
    plt.title("ID3 Decision Tree Performance vs. Maximum Depth", fontsize=13, fontweight="bold")
    plt.xlabel("Max Depth", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(depths)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_overfitting_curve.png"), dpi=300)
    plt.close()

    # Matriz de Confusión (2x2) del clasificador ID3 del Agente Híbrido
    cm = [
        [hybrid_model_metrics["tn"], hybrid_model_metrics["fp"]],
        [hybrid_model_metrics["fn"], hybrid_model_metrics["tp"]],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Safe (0)", "Predicted Mine (1)"])
    ax.set_yticklabels(["Actual Safe (0)", "Actual Mine (1)"])
    ax.set_title("Hybrid Agent Classifier Confusion Matrix", fontsize=12, fontweight="bold")

    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i][j] > (max(cm[0] + cm[1]) / 2) else "black"
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{cm[i][j]}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_confusion_matrix.png"), dpi=300)
    plt.close()

    # Comparación directa de Win Rate: Árbol ID3 vs. Agente Híbrido
    plt.figure(figsize=(7, 5))
    agents = ["ID3 Decision Tree\n(Pure ML)", "Hybrid Agent\n(Logical Deduction + ID3)"]
    win_rates = [tree_stats["win_rate"], hibrido_stats["win_rate"]]
    colors = ["#2b7bba", "#d95f02"]

    bars = plt.bar(agents, win_rates, color=colors, width=0.45)
    plt.title("Minesweeper Win Rate: ID3 Tree vs. Hybrid Agent (1,000 games)", fontsize=12, fontweight="bold")
    plt.ylabel("Win Rate (%)", fontsize=11)
    plt.ylim(0, max(win_rates + [10]) * 1.20)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_win_rate_comparison.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Cargando y enriqueciendo datasets desde data/ (30 características)...")
    X_train, y_train = cargar_csv("data/train.csv")
    X_test, y_test = cargar_csv("data/test.csv")

    depths = [2, 4, 6, 8, 10]
    train_accs, test_accs = [], []
    best_tree = None
    best_test_acc = -1.0

    print("\nEvaluando profundidades del Árbol de Decisión ID3...")
    for d in depths:
        tree = ArbolDecisionID3(max_profundidad=d, min_muestras_division=15)
        tree.fit(X_train, y_train)

        train_preds = tree.predecir(X_train)
        test_preds = tree.predecir(X_test)

        train_m = calcular_metricas(y_train, train_preds)
        test_m = calcular_metricas(y_test, test_preds)

        train_accs.append(train_m["accuracy"])
        test_accs.append(test_m["accuracy"])
        print(
            f"Profundidad {d:2d} -> Train Acc: {train_m['accuracy']*100:.2f}% | "
            f"Test Acc: {test_m['accuracy']*100:.2f}% | F1: {test_m['f1']:.3f}"
        )

        if test_m["accuracy"] > best_test_acc:
            best_test_acc = test_m["accuracy"]
            best_tree = tree

    # Entrenamiento interno del motor ID3 para el Agente Híbrido
    motor_hibrido = BosqueAleatorioID3(
        num_arboles=9,
        max_profundidad=10,
        min_muestras_division=12,
        max_caracteristicas=14,
        semilla=42,
    )
    motor_hibrido.fit(X_train, y_train)
    hybrid_model_metrics = calcular_metricas(y_test, motor_hibrido.predecir(X_test))

    print("\nSimulando 1.000 partidas reales: Árbol ID3 vs. Agente Híbrido...")
    agente_arbol = AgenteBuscaminas(best_tree)
    agente_hibrido = AgenteBuscaminas(motor_hibrido)

    tree_stats = simulador_jugadas(agente=agente_arbol, usar_deduccion=False, num_partidas=1000)
    hibrido_stats = simulador_jugadas(agente=agente_hibrido, usar_deduccion=True, num_partidas=1000)

    print(f"1. Árbol ID3 (Solo ML) -> Win Rate: {tree_stats['win_rate']:5.2f}% | Casillas: {tree_stats['avg_cells']:.1f}/71")
    print(f"2. Agente Híbrido      -> Win Rate: {hibrido_stats['win_rate']:5.2f}% | Casillas: {hibrido_stats['avg_cells']:.1f}/71")

    generar_graficas(
        depths,
        train_accs,
        test_accs,
        hybrid_model_metrics,
        tree_stats,
        hibrido_stats,
        out_dir="report",
    )
    print("\nGráficos guardados exitosamente en la carpeta 'report/'.")