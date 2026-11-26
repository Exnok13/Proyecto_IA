import csv
import os
import random
import matplotlib.pyplot as plt

from minesweeper import minasweeper

# Cuando Martín suba su código, descomenta la siguiente línea y borra DummyDecisionTree:
# from decision_tree import DecisionTree


class DummyDecisionTree:
    """
    Modelo provisional para probar evaluate.py y los gráficos antes de
    conectar el Árbol de Decisión real de Martín.
    """
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        pass

    def predict_proba(self, x):
        # Heurística simple: si ve números bajos (0 o 1), asume bajo riesgo de mina
        revealed_numbers = [val for val in x if val >= 0]
        if not revealed_numbers:
            return 0.5
        avg_danger = sum(revealed_numbers) / (len(revealed_numbers) * 4.0)
        return min(max(avg_danger, 0.05), 0.95)

# Cargar archivos csv de entrenamiento y prueba
def cargar_csv (filepath: str):
    
    X, y = [], []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        #quitar la primera fila de encabezados
        next(reader) 
        for row in reader:
            X.append([int(val) for val in row[:-1]])
            y.append(int(row[-1]))
    return X, y


def calcular_metricas(y_true: list, y_pred: list) -> dict:
    """Calcula Accuracy, Precision, Recall, F1-Score y Matriz de Confusión."""
    # contat casillas verdaderas positivas, verdaderas negativas, falsas positivas y falsas negativas
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    #Porcentaje de aciertos, precision en descubrir bombas, conjunto de minas reales que se encontraron y resumen de aciertos y errores
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

# Metodo que simula partidas de Buscaminas y devuelve estadísticas de desempeño del agente.
def simulador_jugadas(agent_type: str, model=None, num_juegos: int = 1000, seed: int = 42) -> dict:
    random.seed(seed)
    juego = minasweeper(filas=9, columnas=9, can_minas=10)
    ganadas = 0
    total_reveladas = 0

    for _ in range(num_juegos):
        juego.reset()
        f, c = random.randint(0, juego.filas - 1), random.randint(0, juego.columnas - 1)
        
        over, win = juego.jugada(f, c)

        while not over:
            frontera = juego.busqueda_casilla()
            if not frontera:
                casillas_ocultas = [
                    (ff, cc)
                    for ff in range(juego.filas)
                    for cc in range(juego.columnas)
                    if not juego.visible[ff][cc]
                ]
                if not casillas_ocultas:
                    break
                siguiente_f, siguiente_c = random.choice(casillas_ocultas)
            else:
                if agent_type == "random":
                    siguiente_f, siguiente_c = random.choice(frontera)
                else:
                    # Elegir la casilla frontera con menor probabilidad de mina
                    best_cell = None
                    lowest_risk = float("inf")
                    for fr, fc in frontera:
                        window = juego.ventana_5x5(fr, fc)
                        risk = model.predict_proba(window)
                        if risk < lowest_risk:
                            lowest_risk = risk
                            best_cell = (fr, fc)
                    siguiente_f, siguiente_c = best_cell

            over, win = juego.jugada(siguiente_f, siguiente_c)

        if win:
            ganadas += 1
        total_reveladas += juego.casilla_descubiertas

    return {
        "win_rate": (ganadas / num_juegos) * 100,
        "avg_revealed": total_reveladas / num_juegos,
    }

# Metodo que genera y guarda los gráficos de desempeño del Árbol de Decisión y la comparación con el agente aleatorio.
def generar_graficas(depths, train_accs, test_accs, best_metrics, random_stats, tree_stats, out_dir="report"):
    #Genera la carpeta report si no existe
    os.makedirs(out_dir, exist_ok=True)

    # dibujar lineas comparando la exactitud de entrenamiento y prueba en función de la profundidad máxima del árbol
    plt.figure(figsize=(8, 5))
    plt.plot(depths, [a * 100 for a in train_accs], marker="o", label="Training Accuracy", linewidth=2)
    plt.plot(depths, [a * 100 for a in test_accs], marker="s", label="Test Accuracy", linewidth=2)
    plt.title("Decision Tree Performance vs. Maximum Depth", fontsize=13, fontweight="bold")
    plt.xlabel("Max Depth", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(depths)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_overfitting_curve.png"), dpi=300)
    plt.close()

    # grafica de matriz de confusión para el mejor modelo en el conjunto de prueba
    cm = [
        [best_metrics["tn"], best_metrics["fp"]],
        [best_metrics["fn"], best_metrics["tp"]],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Safe (0)", "Predicted Mine (1)"])
    ax.set_yticklabels(["Actual Safe (0)", "Actual Mine (1)"])
    ax.set_title("Confusion Matrix on Unseen Test Set", fontsize=13, fontweight="bold")

    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i][j] > (max(cm[0] + cm[1]) / 2) else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i][j]}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_confusion_matrix.png"), dpi=300)
    plt.close()

    # comparación de tasas de victoria entre el agente aleatorio y el Árbol de Decisión en 1,000 juegos simulados
    plt.figure(figsize=(7, 5))
    agents = ["Random Baseline", "Decision Tree Agent"]
    win_rates = [random_stats["win_rate"], tree_stats["win_rate"]]
    colors = ["#8c9ba5", "#2b7bba"]

    bars = plt.bar(agents, win_rates, color=colors, width=0.5)
    plt.title("Minesweeper Win Rate Comparison (1,000 juegos)", fontsize=13, fontweight="bold")
    plt.ylabel("Win Rate (%)", fontsize=11)
    plt.ylim(0, max(win_rates + [10]) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.8, f"{height:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_win_rate_comparison.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Cargando datasets desde data/...")
    X_train, y_train = cargar_csv ("data/train.csv")
    X_test, y_test = cargar_csv ("data/test.csv")

    depths = [2, 4, 6, 8, 10, 12]
    train_accs, test_accs = [], []
    best_tree = None
    best_test_acc = -1.0
    best_metrics = None

    print("\nEvaluando profundidades del Árbol de Decisión...")
    for d in depths:
        # Cambiar DummyDecisionTree por DecisionTree cuando Martín lo tenga listo
        tree = DummyDecisionTree(max_depth=d)
        tree.fit(X_train, y_train)

        train_preds = [1 if tree.predict_proba(x) >= 0.5 else 0 for x in X_train]
        test_preds = [1 if tree.predict_proba(x) >= 0.5 else 0 for x in X_test]

        train_m = calcular_metricas(y_train, train_preds)
        test_m = calcular_metricas(y_test, test_preds)

        train_accs.append(train_m["accuracy"])
        test_accs.append(test_m["accuracy"])
        print(f"Profundidad {d:2d} -> Train Acc: {train_m['accuracy']*100:.2f}% | Test Acc: {test_m['accuracy']*100:.2f}% | F1: {test_m['f1']:.3f}")

        if test_m["accuracy"] > best_test_acc:
            best_test_acc = test_m["accuracy"]
            best_tree = tree
            best_metrics = test_m

    print("\nSimulando 1.000 partidas reales para comparar Win Rate...")
    random_stats = simulador_jugadas(agent_type="random", num_juegos=1000, seed=42)
    tree_stats = simulador_jugadas(agent_type="tree", model=best_tree, num_juegos=1000, seed=42)

    print(f"Agente Aleatorio    -> Win Rate: {random_stats['win_rate']:.2f}% | Casillas abiertas prom.: {random_stats['avg_revealed']:.1f}")
    print(f"Árbol de Decisión   -> Win Rate: {tree_stats['win_rate']:.2f}% | Casillas abiertas prom.: {tree_stats['avg_revealed']:.1f}")

    generar_graficas(depths, train_accs, test_accs, best_metrics, random_stats, tree_stats, out_dir="report")
    print("\nGráficos guardados exitosamente en la carpeta 'report/'.")