
# Código de simulación de vehículos autónomos
# Autores: 
# Laurie Hernández
# Emilio González
# Hugo Gamboa
# Alejandro Corral 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

# parametros globales. 
alpha = 0.5 # sensibilidad al espaciamiento 
beta = 0.8 # sensibilidad a la velocidad relativa 
delta = 5.0 # distancia mínima 
gamma = 0.5 # distancia dinámica 
dinicial = 10 # distancia inicial entre los carritos
N = 5 # no. inicial de carritos 
t0 = 0 # tiempo inicial de la sim. 
tf = 30 # tiempo final de la sim. 
h = 0.01 # paso h en RK4
t_leader = 5  # tiempo en que acelera el líder

# parametros modificables
def establecer_params(): 
    global alpha, beta, delta, gamma, dinicial
    alpha = float(input("Ingresa sensibilidad al espaciamiento: "))
    beta = float(input("Ingresa sensibilidad a la velocidad relativa: "))
    delta = float(input("Ingresa distancia mínima entre vehículos: "))
    gamma = float(input("Ingresa distancia dinámica: "))
    dinicial = float(input("Ingresa la distancia inicial entre vehículos: "))

def num_vehic():
    global N
    N = int(input("Ingresa el número de vehículos: "))


def tiempo_paso(): 
    global t0, tf, h
    t0 = float(input("Ingresa tiempo inicial: "))
    tf = float(input("Ingresa tiempo final: "))
    h = float(input("Ingresa paso para RK4: "))

#ac. del lider
def leader_accel(t):
    if t < t_leader:
        return 1.0
    return 0.0

# EDs 
def f(t, Y):
    global alpha, beta, delta, gamma, N
    dY = np.zeros_like(Y)

    for i in range(N):
        xi = Y[2*i]
        vi = Y[2*i + 1]

        # dx/dt = v
        dY[2*i] = vi

        if i == 0:
            dY[2*i + 1] = leader_accel(t)
        else:
            x_prev = Y[2*(i-1)]
            v_prev = Y[2*(i-1) + 1]

            s = delta + gamma * vi

            dY[2*i + 1] = alpha * (x_prev - xi - s) + beta * (v_prev - vi)

    return dY


# runge kuta de 4to orden 
def rk4_step(fun, t, Y, h):
    k1 = h * fun(t, Y)
    k2 = h * fun(t + h/2, Y + k1/2)
    k3 = h * fun(t + h/2, Y + k2/2)
    k4 = h * fun(t + h, Y + k3)
    return Y + (k1 + 2*k2 + 2*k3 + k4) / 6


# simulación 
def simular(): 
    global N, t0, tf, h, dinicial

    steps = int((tf - t0) / h)
    Y = np.zeros(2*N)

    # condiciones iniciales
    for i in range(N):
        Y[2*i] = -i * dinicial
        Y[2*i + 1] = 0.0

    history = np.zeros((steps, 2*N))
    times = np.linspace(t0, tf, steps)

    for j in range(steps):
        history[j, :] = Y
        Y = rk4_step(f, times[j], Y, h)

    # paleta de colores pastel
    colors = sns.color_palette("pastel", N)

    # graph de pos. 
    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.plot(times, history[:, 2*i], color=colors[i], label=f"Auto {i+1}")
    plt.ylabel("Posición (m)")
    plt.xlabel("Tiempo (s)")
    plt.title("Posición vs tiempo")
    plt.legend()
    plt.grid()
    plt.show()

    # grafica de velocidad. 
    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.plot(times, history[:, 2*i + 1], color=colors[i], label=f"Auto {i+1}")
    plt.ylabel("Velocidad (m/s)")
    plt.xlabel("Tiempo (s)")
    plt.title("Velocidad vs tiempo")
    plt.legend()
    plt.grid()
    plt.show()
    
    return history, times

def detectar_choque(history, tol=0.1):
    # detector de choques. Regresa True y demás si hay choque entre autos i y j en un frame. 
    # tol es la distancia mínima para considerarse un choque. 
    frames = history.shape[0]
    N_autos = history.shape[1] // 2

    for f in range(frames):
        posiciones = history[f, 0::2]  # x0, x1, x2, ...
        for i in range(N_autos):
            for j in range(i+1, N_autos):
                if abs(posiciones[i] - posiciones[j]) < tol:
                    return True, f, i, j
    return False, None, None, None

# como soy increíble, voy a intentar hacer una simulación 2d de cómo se ven los carritos. 
def animar(history, times):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_xlim(np.min(history[:,::2]) - 5, np.max(history[:,::2]) + 5)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Posición (m)")
    ax.set_yticks([])
    ax.set_title("Animación 2D de vehículos")

    # Creamos un objeto tipo "scatter" para cada auto
    puntos = []
    colors = sns.color_palette("pastel", N)
    for i in range(N):
        punto, = ax.plot([], [], "o", color=colors[i], markersize=10)
        puntos.append(punto)

    # actualización de cada frame
    def update(frame):
        for i in range(N):
            x = history[frame, 2*i] # posición del auto i
            puntos[i].set_data([x], [0])
        return puntos

    ani = animation.FuncAnimation(
        fig, update, frames=len(times), interval=20, blit=True)
    ani._resize_id = None   
    plt.show()

def main():
    global t_leader

    option = 0
    while option != 7:
        print("\n---------- Simulador de Vehículos Autónomos ----------")
        print(f"1. Correr simulación\n   α={alpha}, β={beta}, δ={delta}, γ={gamma}, d0={dinicial}")
        print(f"2. Cambiar parámetros")
        print(f"3. Cambiar número de vehículos (actual={N})")
        print(f"4. Cambiar tiempo y paso (t0={t0}, tf={tf}, h={h})")
        print(f"5. Cambiar tiempo de aceleración del líder (actual={t_leader})")
        print("6. Visualizar simulación 2D de los carros")
        print("7. Salir")
        option = int(input("Opción: "))

        if option == 1:
            simular()
        elif option == 2:
            establecer_params()
        elif option == 3:
            num_vehic()
        elif option == 4:
            tiempo_paso()
        elif option == 5:
            t_leader = float(input("Ingresa nuevo tiempo de aceleración del líder: "))
        elif option == 6: 
            history, times = simular()
            # detectamos si hay un choque antes de animar. 
            choque, frame, i, j = detectar_choque(history)
            if (choque):
                print(f"Choque detectado entre Auto {i+1} y Auto {j+1} en t = {times[frame]:.2f} s")
                print("¿Deseas continuar y visualizar?\n1. Sí\n2. No")
                n = int(input())
                if(n == 1): 
                    animar(history, times)
            else:
                animar(history, times)
        else:
            print("Gracias por usar el simulador <3")

main()
