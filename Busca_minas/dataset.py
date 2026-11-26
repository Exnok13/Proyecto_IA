import csv
import os
import random
from minasweeper import minasweeper
    
# Metodo que simula partidas para la recolección de datos de entrenamiento para el modelo de Buscaminas.
def collect_raw_muestras(can_jugadas: int = 5000, semilla: int = 42) -> list:
    
    random.seed(semilla)
    juego = minasweeper(filas=9, columnas=9, can_minas=10)  
    # dicionario para guardar las ventanas 5x5
    diccionario = {}  

    for _ in range(can_jugadas):
        juego.reset()
        # Primer movimiento aleatorio en el tablero
        f, c = random.randint(0, juego.filas - 1), random.randint(0, juego.columnas - 1)
        over, _ = juego.jugada(f, c)

        while not over:
            frontera = juego.busqueda_casilla()
            if not frontera:
                # Si no hay frontera visible, elegir una casilla oculta al azar
                casilla_oculta = [
                    (ff, cc)
                    for ff in range(juego.filas)
                    for cc in range(juego.columnas)
                    if not juego.visible[ff][cc]
                ]   
                if not casilla_oculta:
                    break
                siguiente_f, siguiente_c = random.choice(casilla_oculta)
            else:
                # Extraer todas las ventanas de la frontera actual antes de mover
                for fr, fc in frontera:
                    ventana = tuple(juego.ventana_5x5(fr, fc))
                    label = 1 if juego.tablero[fr][fc] == -1 else 0
                    diccionario[ventana] = label

                # Elegir una casilla frontera al azar para avanzar la partida
                siguiente_f, siguiente_c = random.choice(frontera)

            over, _ = juego.jugada(siguiente_f, siguiente_c)

    # Convertir el diccionario a una lista de (features_list, label)
    dataset = [(list(ventana), label) for ventana, label in diccionario.items()]
    return dataset

# Equilibrar las jugadas en 50% minas y 50% seguras mediante submuestreo y divide los datos entre un 80% entrenamiento 
# y un 20% prueba.
def balance_and_split(dataset: list, train_ratio: float = 0.8, semilla: int = 42):
    random.seed(semilla)

    minas = [muestra for muestra in dataset if muestra[1] == 1]
    seguro = [muestra for muestra in dataset if muestra[1] == 0]

    print(f"Muestras únicas recolectadas -> Seguras (0): {len(seguro)} | Minas (1): {len(minas)}")

    # Igualar las muestras mayoritaria al tamaño de la minoritaria
    muestra_minima = min(len(minas), len(seguro))
    balanceo_mina = random.sample(minas, muestra_minima)
    balanceo_seguro = random.sample(seguro, muestra_minima)

    balanceo_dataset = balanceo_mina + balanceo_seguro
    random.shuffle(balanceo_dataset)

    split_idx = int(len(balanceo_dataset) * train_ratio)
    train_data = balanceo_dataset[:split_idx]
    test_data = balanceo_dataset[split_idx:]

    return train_data, test_data

# Metodo que guarda las muestras en un archivo CSV con encabezados f0..f23 y label.
def save_to_csv(data: list, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = [f"f{i}" for i in range(24)] + ["label"]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for features, label in data:
            writer.writerow(features + [label])


if __name__ == "__main__":
    print("Simulando 5.000 partidas para extraer ventanas 5x5...")
    raw_data = collect_raw_muestras(can_jugadas=5000, semilla=42)

    train_set, test_set = balance_and_split(raw_data, train_ratio=0.8, semilla=42)

    save_to_csv(train_set, "data/train.csv")
    save_to_csv(test_set, "data/test.csv")

    print(f"Dataset balanceado guardado exitosamente:")
    print(f" - Entrenamiento (data/train.csv): {len(train_set)} filas")
    print(f" - Prueba        (data/test.csv) : {len(test_set)} filas")