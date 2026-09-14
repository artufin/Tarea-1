# Tarea 1 — Bootstrapping paralelo para regresión lineal

**IIC3533 · Computación de Alto Rendimiento · 2026-2**
Entrega: viernes 25 de septiembre de 2026, 23:59 (Canvas, documento PDF).

Este repositorio contiene las tres implementaciones del bootstrap pedidas en el ítem (b),
más los scripts de medición y graficado necesarios para armar el informe.

Parámetros del experimento (fijos por enunciado): `N = 100.000`, `k = 300`, `B = 48`.

---

## 1. Requisitos

- Python 3.13
- Los paquetes de `requirements.txt`: `numpy`, `matplotlib`, `joblib`, `threadpoolctl`, `scikit-learn`

> `scikit-learn` no aparece en el comando de instalación del enunciado, pero el ítem (b) lo
> necesita (`BaggingRegressor` y `LinearRegression`). Ya está incluido en `requirements.txt`.

**Espacio y memoria:** la matriz `X` es de 100.000 × 301 en float64 ≈ **241 MB**. Se cachea en
disco (`datos/`) y cada proceso worker necesita acceso a ella, así que conviene tener al menos
4 GB de RAM libres y ~300 MB de disco.

## 2. Instalación

```bash
pip install -r requirements.txt
```

Si prefieren aislar el entorno (recomendado por el enunciado, opcional aquí):

```bash
conda create -n tarea1-hpc python=3.13 -y
conda activate tarea1-hpc
conda install numpy matplotlib joblib threadpoolctl scikit-learn -y
```

## 3. Estructura del proyecto

| Archivo | Qué hace | Ítems |
|---|---|---|
| `common.py` | Genera y cachea los datos sintéticos, calcula los intervalos de confianza, cronometra y escribe el CSV | (a) |
| `bs_auto.py` | `BaggingRegressor` con `n_jobs=p` (bootstrap y paralelismo internos) | (b) |
| `bs_sklearn.py` | `joblib.Parallel` + `LinearRegression` | (b) |
| `bs_numpy.py` | `joblib.Parallel` + NumPy puro; además inspecciona los threads de BLAS | (b), (e) |
| `bench.py` | Driver de medición: barre `p`, barre la grilla `(p, t)` y escribe `resultados/tiempos.csv` | (f), (i) |
| `check.py` | Compara los intervalos de las tres versiones y verifica reproducibilidad | (c) |
| `plots.py` | Genera todas las figuras en `figuras/` | (f), (g), (h), (i), (j) |

Los ítems **(d)** y **(j)** son de redacción: (d) no requiere código, y (j) se responde
analizando los CSV de ambos computadores.

## 4. Orden de ejecución

Ejecutar **en este orden**, desde la raíz del proyecto. Los pasos 1 a 6 deben repetirse
**en los dos computadores** (el enunciado exige experimentos en al menos dos máquinas).

```bash
# 1. Generar y cachear los datos sintéticos (ítem a). Solo la primera vez.
python common.py

# 2. Verificar correctitud y reproducibilidad (ítem c).
#    Debe correrse ANTES de medir tiempos: no tiene sentido optimizar algo que da mal.
python check.py

# 3. Comparar las variantes de bs_numpy para documentar las mejoras (ítem b).
python bench.py --variantes

# 4. Diagnosticar oversubscription de threads (ítem e).
#    Correr en paralelo con el Administrador de tareas / htop abierto.
python bs_numpy.py --threads-info -p 4

# 5. Barrido principal de tiempos: 3 versiones × p ∈ {1..p_máx} (ítem f).
#    Es el paso más lento: varios minutos.
python bench.py --barrido

# 6. Grilla combinada de procesos y threads, con p·t ≤ p_máx (ítem i).
python bench.py --grilla

# 7. Generar las figuras de esta máquina (ítems f, g, h, i).
python plots.py
```

Una vez que **ambos** computadores hayan corrido los pasos 1–6, juntar los dos
`resultados/tiempos.csv` en uno solo (las filas ya vienen etiquetadas con la columna
`maquina`) y generar la comparación final:

```bash
# 8. Comparación entre los dos equipos (ítem j).
python plots.py --comparar
```

### Antes de medir

Las mediciones de los pasos 3, 5 y 6 son sensibles al ruido del sistema. Para que los tiempos
sean comparables entre máquinas:

- Cerrar navegador, Docker, IDEs y cualquier cosa que consuma CPU.
- **Conectar el notebook a la corriente.** Con batería, Windows y Linux bajan la frecuencia del
  CPU y los tiempos quedan inflados de forma irregular.
- Correr cada configuración varias veces y reportar la **mediana**, no el promedio: en notebooks
  el throttling térmico produce outliers hacia arriba que arrastran el promedio.

### Ficha de cada máquina (completar para el informe)

El ítem (f) pide declarar explícitamente el número de cores lógicos y el ítem (j) necesita estos
datos para justificar las diferencias. `python common.py` los imprime.

| Máquina | CPU | Cores físicos / lógicos | RAM | SO | Backend BLAS |
|---|---|---|---|---|---|
| (Arturo) | Intel i7-6600U @ 2.60 GHz | 2 / 4 | 16 GB | Windows 10 Pro | (completar) |
| (2da máquina) | | | | | |

---

## 5. Advertencia importante: Windows no tiene `fork`

**Esto afecta directamente la respuesta del ítem (d)**, que pregunta cómo el backend
`multiprocessing` de joblib crea los procesos, cómo se asigna memoria y qué ocurre con los
arreglos `X` e `y` al lanzar `p` procesos. La respuesta **no es la misma en Windows que en
Linux/macOS**, y el enunciado está redactado asumiendo Linux/macOS.

### Qué cambia

| | Linux / macOS | Windows |
|---|---|---|
| Creación de procesos | `fork()`: el hijo es una copia del padre | `spawn`: se lanza un intérprete nuevo y vacío |
| Memoria al crear el proceso | *Copy-on-write*: `X` no se copia; padre e hijo comparten las mismas páginas físicas hasta que alguien escriba | El hijo arranca sin nada; `X` debe transferirse explícitamente |
| Costo de arrancar un worker | Bajo (milisegundos) | Alto: re-importa el intérprete, NumPy y sklearn en cada hijo |
| Guard `if __name__ == "__main__"` | Recomendable | **Obligatorio** |

### Consecuencias prácticas

1. **El guard es obligatorio.** Con `spawn`, cada proceso hijo vuelve a importar el módulo que lo
   lanzó. Sin `if __name__ == "__main__":` cada hijo relanzaría el experimento completo, generando
   procesos infinitos hasta colgar la máquina. Todos los scripts de este repo ya lo tienen.

2. **`X` no se comparte por copy-on-write.** joblib compensa esto con *memory-mapping* automático:
   los arreglos de más de `max_nbytes` (1 MB por defecto) se vuelcan a un archivo temporal `.npy`
   y cada worker lo abre como `np.memmap` en modo lectura. O sea, en Windows los 241 MB pasan por
   disco una vez, en vez de compartirse en RAM como haría el COW de Linux. **Vale la pena
   mencionarlo explícitamente en el ítem (d)** y verificarlo mirando el directorio temporal de
   joblib mientras corre.

3. **El overhead del ítem (h) será mayor en Windows.** Arrancar `p` intérpretes nuevos e importar
   NumPy/sklearn en cada uno cuesta del orden de cientos de milisegundos por worker. Si una de las
   dos máquinas del grupo es Linux y la otra Windows, **esta es probablemente la diferencia más
   notable que van a observar en el ítem (j)**, y es una explicación sólida para justificarla.

4. **El monitor de actividad es otro.** El enunciado nombra el Monitor de Actividad (macOS) y el
   Monitor del Sistema (Linux). En Windows el equivalente es el **Administrador de tareas**
   (pestaña Rendimiento) o, para ver uso por proceso con más detalle, el **Monitor de recursos**
   (`resmon.exe`).

5. **Ojo con el backend por defecto.** joblib usa `loky` por defecto, que evita `fork` *incluso en
   Linux* (deliberadamente, para no mezclar `fork` con los threads de OpenMP/BLAS, combinación que
   puede producir deadlocks). Para observar el comportamiento de `fork` que describe el ítem (d)
   hay que pedirlo explícitamente:

   ```python
   Parallel(n_jobs=p, backend="multiprocessing")(...)   # en Linux: fork + copy-on-write
   ```

   Comparar `loky` contra `multiprocessing` en Linux es un experimento corto que da material
   directo para el ítem (d).

### Alternativa en Linux

Si quieren observar empíricamente el comportamiento de `fork` y copy-on-write:

- **Opción A — que la segunda máquina del grupo sea Linux.** Es lo ideal: cubre el requisito de los
  dos computadores del ítem (j) *y* permite contrastar `fork` contra `spawn` en el ítem (d).
- **Opción B — WSL2 sobre el mismo Windows** (disponible en Windows 10 Pro):

  ```bash
  wsl --install -d Ubuntu     # desde PowerShell como administrador
  ```

  Dentro de la distro, instalar Python 3.13 y `pip install -r requirements.txt`, y correr los
  mismos comandos.

  > **Importante:** WSL2 corre en una VM con su propio scheduler y su propia asignación de memoria,
  > así que **sirve para observar `fork`/COW en el ítem (d), pero NO reemplaza al segundo
  > computador del ítem (j)** — los tiempos no son comparables con los del host, y seguiría siendo
  > el mismo hardware.

Para verificar en Linux qué método de arranque está en uso:

```python
import multiprocessing as mp
print(mp.get_start_method())   # 'fork' en Linux, 'spawn' en Windows
```

Y para ver el efecto del copy-on-write: lanzar el experimento y comparar en `htop` la memoria
**RSS** de cada worker contra la memoria total del sistema. Con `fork`, la suma de los RSS excede
por mucho la memoria realmente usada, porque las páginas de `X` están compartidas.

---

## 6. Notas para el informe

- **Declarar el uso de IA.** El enunciado lo permite explícitamente, pero exige que quede
  declarado en el informe.
- **Gráficos y tablas se evalúan.** Títulos, labels de ejes con unidades, leyendas y tamaño de
  fuente legible. `plots.py` ya aplica un estilo común a todas las figuras.
- **Justificar la elección de `T(1)`** en el ítem (g): usar el `T(1)` de cada versión mide la
  escalabilidad propia de cada una, pero premia a una versión que es lenta en serie. Usar el
  `T(1)` de la versión más rápida da el *speedup* absoluto. Conviene mostrar uno y comentar el otro.
- **Los intervalos no van a coincidir bit a bit** entre las tres versiones (ítem c): cada una
  sortea sus índices con un generador distinto. La equivalencia que se pide es estadística —los
  intervalos deben coincidir dentro del error de Monte Carlo de `B = 48` resamples—, no numérica.
