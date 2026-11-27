import random
from collections import deque

class minasweeper:
    
    
    #Simulador de busca minas para entrenamiento de agentes de IA.

    # Creación del tablero y estado inicial
    def __init__(self, filas: int = 9, columnas: int = 9, can_minas: int = 10):
        self.filas = filas
        self.columnas = columnas
        self.can_minas = can_minas
        self.reset()

    
    # Reinicia el juego y coloca las minas desde el inicio, para crear multiples partidas de entrenamiento
    def reset(self):
        """
        Metodo que reinicia el juego, colocando minas aleatoriamente y reseteando el estado del tablero y las banderas de victoria/derrota.
        """
        # -1 representa mina, 0-8 representa cantidad de minas adyacentes
        self.tablero = [[0 for _ in range(self.columnas)] for _ in range(self.filas)]
        # True si la casilla fue descubierta, False si sigue oculta
        self.visible = [[False for _ in range(self.columnas)] for _ in range(self.filas)]
        # Banderas de estado del juego
        self.perdiste = False
        self.ganaste = False
        #contador de casillas descubiertas para verificar victoria
        self.casilla_descubiertas = 0
        
        # colocar minas aleatoriamente en el tablero
        self.colocar_minas()

    def limite(self, fila: int, columna: int) -> bool:
        #Verificar si las coordenadas están dentro de los límites del tablero.
        return 0 <= fila < self.filas and 0 <= columna < self.columnas
    
    # Metodo que revisa los vecinos de una casilla y devuelve ua lista de coordenadas de hasta 8 vecinos adyacnetes
    def _get_vecinos(self, fila: int, columna: int) -> list:
        
        vecinos = []
        # revisa todos los vecionas de la casilla y los agrega a una lista
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = fila + dr, columna + dc
                if self.limite(nr, nc):
                    vecinos.append((nr, nc))
        return vecinos
    
    # Metodo que coloca las minas en el tablero y calcula los numeros de las casillas adyacentes
    def colocar_minas(self):
        all_cells = [
            (f, c)
            for f in range(self.filas)
            for c in range(self.columnas)
        ]

        minas = random.sample(all_cells, self.can_minas)
        for f, c in minas:
            self.tablero[f][c] = -1

        # Calcular los números (0 a 8) para el resto de las casillas
        for f in range(self.filas):
            for c in range(self.columnas):
                if self.tablero[f][c] == -1:
                    continue
                contador_minas = sum(
                    1 for nr, nc in self._get_vecinos(f, c)
                    if self.tablero[nr][nc] == -1
                )
                self.tablero[f][c] = contador_minas
    #Metodo que procesa una jugada del jugador, revelando la casilla y propagando si es necesario.
    def jugada(self, f: int, c: int) -> tuple:
        # Retorna una tupla (perdiste, ganaste) indicando si el juego terminó y si se ganó.
        if self.perdiste or not self.limite(f, c) or self.visible[f][c]:
            return self.perdiste, self.ganaste

        # Pisó una mina 
        if self.tablero[f][c] == -1:
            self.visible[f][c] = True
            self.perdiste = True
            self.ganaste = False
            return True, False

        # Casilla segura, se expande en cascada (BFS) si es 0
        self.revelar_casilla(f, c)

        # Verificar condición de victoria
        celdas_seguras = (self.filas * self.columnas) - self.can_minas
        if self.casilla_descubiertas == celdas_seguras:
            self.perdiste = True
            self.ganaste = True

        return self.perdiste, self.ganaste
    
    # Metodo que revela una casilla y propaga automáticamente si encuentra 0
    def revelar_casilla(self, inicio_f: int, inicio_c: int):
        
        queue = deque([(inicio_f, inicio_c)])
        self.visible[inicio_f][inicio_c] = True
        self.casilla_descubiertas += 1

        while queue:
            r, c = queue.popleft()
            if self.tablero[r][c] == 0:
                for nr, nc in self._get_vecinos(r, c):
                    if not self.visible[nr][nc] and self.tablero[nr][nc] != -1:
                        self.visible[nr][nc] = True
                        self.casilla_descubiertas += 1
                        queue.append((nr, nc))
    # Metodo que devuelve todas las casillas ocultas que tienen al menos un vecino descubierto
    #Esto para encontrar las casillas para que el agente pueda tomar decisiones de jugadas. 
    def busqueda_casilla(self) -> list:
        
        frontier = []
        for f in range(self.filas):
            for c in range(self.columnas):
                if not self.visible[f][c]:
                    if any(self.visible[nr][nc] for nr, nc in self._get_vecinos(f, c)):
                        frontier.append((f, c))
        return frontier
    # Obtienen los valores de las casillas alrededor de la casilla (r, c) en una ventana de 5x5, excluyendo la casilla central.
    def ventana_5x5(self, f: int, c: int) -> list:
        
        #-2 = fuera del tablero
        #-1 = casilla oculta
        # 0 a 8 = número descubierto
        
        ventana = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue  # Excluir la casilla del medio
                nr, nc = f + dr, c + dc
                if not self.limite(nr, nc):
                    ventana.append(-2)
                elif not self.visible[nr][nc]:
                    ventana.append(-1)
                else:
                    ventana.append(self.tablero[nr][nc])
        return ventana
    # Imprime el tablero 
    def print_tablero(self):
    
        print("  " + " ".join(str(c) for c in range(self.columnas)))
        for f in range(self.filas):
            filas = []
            for c in range(self.columnas):
                if not self.visible[f][c]:
                    filas.append("■")
                elif self.tablero[f][c] == -1:
                    filas.append("*")
                else:
                    filas.append(str(self.tablero[f][c]))
            print(f"{f} " + " ".join(filas))


# --- Bloque de prueba rápida ---
if __name__ == "__main__":
    juego = minasweeper(filas=9, columnas=9, can_minas=10)
    # Primer clic 
    over, ganaste = juego.jugada(4, 4)
    juego.print_tablero()
    
    frontier = juego.busqueda_casilla()
    print(f"\nCasillas frontera detectadas: {len(frontier)}")
    if frontier:
        sample_r, sample_c = frontier[0]
        ganaste = juego.ventana_5x5(sample_r, sample_c)
        label = 1 if juego.tablero[sample_r][sample_c] == -1 else 0
        print(f"Ejemplo ventana 5x5 para ({sample_r}, {sample_c}) -> Largo: {len(ganaste)}, Etiqueta real: {label}")
        print(ganaste)