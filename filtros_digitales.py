"""Diseño y evaluación de filtros digitales pasa bajos, pasa altos y pasa bandas.

El experimento usa una señal compuesta de tonos conocidos y ruido blanco
reproducible. Los filtros se aplican hacia adelante y hacia atrás para evitar
el retardo de fase en esta demostración fuera de línea.
"""
from pathlib import Path
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

BASE = Path(__file__).resolve().parent
FIGURES = BASE / "figuras"
RESULTS = BASE / "resultados"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Parámetros de muestreo y duración del registro.
FS = 1000.0  # muestras por segundo (Hz)
DURATION = 2.0  # segundos
N = int(FS * DURATION)
t = np.arange(N) / FS
RNG = np.random.default_rng(2026)  # semilla fija para repetir el experimento

# La señal útil contiene 5, 50 y 120 Hz. Se suma ruido blanco gaussiano.
components = {
    5.0: 1.00,
    50.0: 0.70,
    120.0: 0.45,
}
clean = sum(amplitude * np.sin(2 * np.pi * frequency * t)
            for frequency, amplitude in components.items())
noise = 0.45 * RNG.standard_normal(N)
noisy = clean + noise

# IIR: Butterworth pasa bajos en 20 Hz; Chebyshev I pasa altos en 80 Hz.
low_sos = signal.butter(6, 20, btype="lowpass", fs=FS, output="sos")
high_sos = signal.cheby1(5, 0.5, 80, btype="highpass", fs=FS, output="sos")

# FIR: pasa bandas de 40 a 60 Hz diseñado con ventana Hamming.
band_taps = signal.firwin(121, [40, 60], pass_zero=False,
                          window="hamming", fs=FS)

# filtfilt/sosfiltfilt se usan para que los gráficos comparen las formas
# sin el retardo de fase que introduce una aplicación causal de una pasada.
filtered = {
    "Pasa bajos · Butterworth IIR": signal.sosfiltfilt(low_sos, noisy),
    "Pasa altos · Chebyshev I IIR": signal.sosfiltfilt(high_sos, noisy),
    "Pasa bandas · FIR Hamming": signal.filtfilt(band_taps, [1.0], noisy),
}

filters = [
    ("Pasa bajos · Butterworth IIR", "low", low_sos, "IIR · Butterworth, orden 6 · fc = 20 Hz"),
    ("Pasa altos · Chebyshev I IIR", "high", high_sos, "IIR · Chebyshev I, orden 5 · fc = 80 Hz"),
    ("Pasa bandas · FIR Hamming", "band", band_taps, "FIR · ventana Hamming, 121 coeficientes · 40–60 Hz"),
]

# Amplitud de cada tono estimada por mínimos cuadrados sobre seno y coseno.
def tone_amplitude(x: np.ndarray, frequency: float) -> float:
    design = np.column_stack((np.sin(2*np.pi*frequency*t),
                              np.cos(2*np.pi*frequency*t),
                              np.ones_like(t)))
    coeff, *_ = np.linalg.lstsq(design, x, rcond=None)
    return float(np.hypot(coeff[0], coeff[1]))

# Resumen cuantitativo: amplitud estimada de los tonos y cambio frente a entrada.
metric_rows = []
for label, y in filtered.items():
    for frequency in components:
        before = tone_amplitude(noisy, frequency)
        after = tone_amplitude(y, frequency)
        db = 20 * np.log10(max(after, 1e-12) / max(before, 1e-12))
        metric_rows.append({
            "filtro": label,
            "frecuencia_hz": frequency,
            "amplitud_entrada": before,
            "amplitud_salida": after,
            "cambio_db": db,
        })

with (RESULTS / "metricas_tonos.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
    writer.writeheader()
    writer.writerows(metric_rows)

rms_values = {"entrada": float(np.sqrt(np.mean(noisy**2)))}
rms_values.update({label: float(np.sqrt(np.mean(y**2))) for label, y in filtered.items()})
(RESULTS / "resumen.json").write_text(json.dumps({
    "frecuencia_muestreo_hz": FS,
    "duracion_s": DURATION,
    "componentes_hz": list(components.keys()),
    "rms": rms_values,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# Estilo uniforme y colores de lectura sencilla.
COLORS = {"entrada": "#64748b", "low": "#0f766e", "high": "#d97706", "band": "#6d28d9"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.22})

# Figura 1: entrada ruidosa y sus componentes conocidas.
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t[:500], noisy[:500], color=COLORS["entrada"], lw=0.9, label="Señal con ruido")
ax.plot(t[:500], clean[:500], color="#0f172a", lw=1.5, alpha=0.8, label="Señal útil original")
ax.set(title="Señal de prueba antes del filtrado", xlabel="Tiempo (s)", ylabel="Amplitud")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(FIGURES / "01_senal_entrada.png", dpi=180)
plt.close(fig)

# Figura 2: respuesta en frecuencia. El filtrado bidireccional equivale a |H|².
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, (label, kind, coeffs, subtitle), color in zip(axes, filters, [COLORS["low"], COLORS["high"], COLORS["band"]]):
    if kind in ("low", "high"):
        f, h = signal.sosfreqz(coeffs, worN=4096, fs=FS)
    else:
        f, h = signal.freqz(coeffs, worN=4096, fs=FS)
    # La operación ida y vuelta aplica dos veces la magnitud del filtro.
    db = 40 * np.log10(np.maximum(np.abs(h), 1e-8))
    ax.plot(f, db, color=color, lw=1.8)
    ax.set_ylim(-100, 5)
    ax.set_ylabel("Ganancia (dB)")
    ax.set_title(subtitle, loc="left", fontsize=10)
    ax.axhline(-3, color="#94a3b8", ls="--", lw=0.8)
    for tone in components:
        ax.axvline(tone, color="#94a3b8", ls=":", lw=0.7)
axes[-1].set_xlim(0, 200)
axes[-1].set_xlabel("Frecuencia (Hz)")
fig.suptitle("Respuesta en frecuencia de los filtros", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "02_respuestas_frecuencia.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figura 3: señal filtrada, junto a la señal útil esperada para cada banda.
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
expected = {
    "Pasa bajos · Butterworth IIR": 1.00 * np.sin(2*np.pi*5*t),
    "Pasa altos · Chebyshev I IIR": 0.45 * np.sin(2*np.pi*120*t),
    "Pasa bandas · FIR Hamming": 0.70 * np.sin(2*np.pi*50*t),
}
for ax, (label, y), color in zip(axes, filtered.items(), [COLORS["low"], COLORS["high"], COLORS["band"]]):
    ax.plot(t[:500], y[:500], color=color, lw=1.0, label="Salida filtrada")
    ax.plot(t[:500], expected[label][:500], color="#111827", lw=1.2, alpha=0.7, label="Componente que se busca conservar")
    ax.set_ylabel("Amplitud")
    ax.set_title(label, loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=2, fontsize=8)
axes[-1].set_xlabel("Tiempo (s)")
fig.suptitle("Comparación temporal: salida y componente objetivo", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "03_comparacion_temporal.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figura 4: espectro de amplitud unilateral de entrada y tres salidas.
def amplitude_spectrum(x: np.ndarray):
    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window)
    freq = np.fft.rfftfreq(len(x), 1/FS)
    amplitude = 2 * np.abs(spectrum) / np.sum(window)
    return freq, amplitude
fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
series = [("Entrada con ruido", noisy, COLORS["entrada"])] + [
    (name, y, color) for (name, y), color in zip(filtered.items(), [COLORS["low"], COLORS["high"], COLORS["band"]])]
for ax, (label, x, color) in zip(axes, series):
    f, a = amplitude_spectrum(x)
    ax.plot(f, a, color=color, lw=1.1)
    ax.set_ylabel("Amplitud")
    ax.set_title(label, loc="left", fontsize=10)
    for tone in components:
        ax.axvline(tone, color="#94a3b8", ls=":", lw=0.7)
axes[-1].set_xlim(0, 180)
axes[-1].set_xlabel("Frecuencia (Hz)")
fig.suptitle("Espectro de amplitud antes y después del filtrado", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "04_espectros.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Experimento completado. Resultados guardados en resultados/ y figuras/.")
print("RMS de entrada:", f"{rms_values['entrada']:.3f}")
for name, value in list(rms_values.items())[1:]:
    print(f"RMS {name}: {value:.3f}")
for row in metric_rows:
    print(f"{row['filtro']} | {row['frecuencia_hz']:.0f} Hz: {row['cambio_db']:.1f} dB")
