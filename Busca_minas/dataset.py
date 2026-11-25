import csv
import os
import random
from minesweeper import minasweeper
    
# Metodo que simula partidas para la recolección de datos de entrenamiento para el modelo de Buscaminas.
def collect_raw_samples(num_games: int = 5000, semilla: int = 42) -> list:
    
    random.seed(semilla)
    game = minasweeper(filas=9, columnas=9, can_minas=10)
    unique_samples = {}  # Usamos un diccionario para evitar ventanas duplicadas

    for _ in range(num_games):
        game.reset()
        # Primer movimiento aleatorio en el tablero
        r, c = random.randint(0, game.filas - 1), random.randint(0, game.columnas - 1)
        over, _ = game.jugada(r, c)

        while not over:
            frontier = game.busqueda_casilla()
            if not frontier:
                # Si no hay frontera visible, elegir una casilla oculta al azar
                hidden = [
                    (rr, cc)
                    for rr in range(game.filas)
                    for cc in range(game.columnas)
                    if not game.visible[rr][cc]
                ]
                if not hidden:
                    break
                next_r, next_c = random.choice(hidden)
            else:
                # Extraer todas las ventanas de la frontera actual antes de mover
                for fr, fc in frontier:
                    window = tuple(game.ventana_5x5(fr, fc))
                    # NOTA: Si en minesweeper.py la matriz se llama self.tablero,
                    # cambia game.tablero[fr][fc] por game.tablero[fr][fc]
                    label = 1 if game.tablero[fr][fc] == -1 else 0
                    unique_samples[window] = label

                # Elegir una casilla frontera al azar para avanzar la partida
                next_r, next_c = random.choice(frontier)

            over, _ = game.jugada(next_r, next_c)

    # Convertir el diccionario a una lista de (features_list, label)
    dataset = [(list(window), label) for window, label in unique_samples.items()]
    return dataset


def balance_and_split(dataset: list, train_ratio: float = 0.8, semilla: int = 42):
    """
    Equilibra las clases (50% minas, 50% seguras) mediante submuestreo (undersampling)
    y divide los datos en conjuntos de entrenamiento (80%) y prueba (20%).
    """
    random.seed(semilla)

    mines = [sample for sample in dataset if sample[1] == 1]
    safe = [sample for sample in dataset if sample[1] == 0]

    print(f"Muestras únicas recolectadas -> Seguras (0): {len(safe)} | Minas (1): {len(mines)}")

    # Igualar la clase mayoritaria al tamaño de la minoritaria
    min_count = min(len(mines), len(safe))
    balanced_mines = random.sample(mines, min_count)
    balanced_safe = random.sample(safe, min_count)

    balanced_dataset = balanced_mines + balanced_safe
    random.shuffle(balanced_dataset)

    split_idx = int(len(balanced_dataset) * train_ratio)
    train_data = balanced_dataset[:split_idx]
    test_data = balanced_dataset[split_idx:]

    return train_data, test_data


def save_to_csv(data: list, filepath: str):
    """Guarda las muestras en un archivo CSV con encabezados f0..f23 y label."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = [f"f{i}" for i in range(24)] + ["label"]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for features, label in data:
            writer.writerow(features + [label])


if __name__ == "__main__":
    print("Simulando 5.000 partidas para extraer ventanas 5x5...")
    raw_data = collect_raw_samples(num_games=5000, semilla=42)

    train_set, test_set = balance_and_split(raw_data, train_ratio=0.8, semilla=42)

    save_to_csv(train_set, "data/train.csv")
    save_to_csv(test_set, "data/test.csv")

    print(f"Dataset balanceado guardado exitosamente:")
    print(f" - Entrenamiento (data/train.csv): {len(train_set)} filas")
    print(f" - Prueba        (data/test.csv) : {len(test_set)} filas")