# Implementación y evaluación de filtros digitales

Proyecto reproducible en Python para diseñar filtros pasa bajos, pasa altos y pasa bandas, aplicarlos a una señal compuesta con ruido blanco y comparar sus respuestas temporales y espectrales.

## Requisitos

- Python 3.10 o posterior
- NumPy, SciPy y Matplotlib

## Instalación y ejecución

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python filtros_digitales.py
```

El programa produce cuatro gráficas en `figuras/` y dos archivos de resultados en `resultados/` (métricas por tono en CSV y resumen RMS en JSON). La semilla aleatoria 2026 permite repetir los mismos datos.

## Diseño utilizado

| Tipo | Diseño | Banda de paso | Uso en la prueba |
|---|---|---:|---|
| Pasa bajos | IIR Butterworth de orden 6 | hasta 20 Hz | conservar 5 Hz y reducir 50 y 120 Hz |
| Pasa altos | IIR Chebyshev I de orden 5, rizado 0.5 dB | desde 80 Hz | conservar 120 Hz y reducir 5 y 50 Hz |
| Pasa bandas | FIR con 121 coeficientes y ventana Hamming | 40–60 Hz | conservar 50 Hz y reducir 5 y 120 Hz |

Los filtros se aplican con `sosfiltfilt` (IIR) o `filtfilt` (FIR) para una comparación sin retardo de fase. Este método realiza filtrado hacia adelante y hacia atrás; por eso la respuesta efectiva en magnitud se eleva al cuadrado. Es apropiado para analizar registros ya capturados, pero no para procesamiento en tiempo real, que requiere una implementación causal.

## Archivos

- `filtros_digitales.py`: generación de la señal, diseño, filtrado, análisis y gráficas.
- `figuras/`: visualizaciones generadas al ejecutar el script.
- `resultados/`: métricas calculadas automáticamente.
- `requirements.txt`: dependencias.

## Publicar en GitHub

1. Crea un repositorio vacío en GitHub.
2. Descomprime este ZIP y abre una terminal dentro de la carpeta.
3. Ejecuta:

```bash
git init
git add .
git commit -m "Implementar filtros digitales y análisis"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE-REPOSITORIO.git
git push -u origin main
```

Reemplaza la dirección de ejemplo por la URL del repositorio que hayas creado.
