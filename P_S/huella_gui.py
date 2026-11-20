#!/usr/bin/env python3
"""
Interfaz gráfica para huella_core.py (reconoce canciones por huella acústica).

Necesita huella_core.py en la MISMA carpeta.
Instalación:  pip install numpy scipy soundfile sounddevice matplotlib
Ejecutar:     python huella_gui.py
"""
import os
import time
import queue
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import spectrogram

import huella_core as fc

DEV_SR = 44100                              # frecuencia de captura del micrófono
NFFT = 1024                                 # puntos de la FFT del espectrograma
FREQS = np.fft.rfftfreq(NFFT, 1 / fc.SR)
COLS = 150                                  # columnas del espectrograma en vivo
FRAME_S = 0.25                              # segundos de señal que se dibujan
TICK_MS = 100                               # refresco de pantalla
DB_MIN, DB_MAX = -110, -10
LIVE_SPAN = COLS * TICK_MS / 1000           # segundos visibles en el espectrograma
LIVE_MIN_S = 5                              # segundos mínimos antes de intentar reconocer
RETRY_S = 2                                 # cada cuántos segundos reintenta
WINDOW_S = 20                               # usa como máximo los últimos 20 s grabados
FILE_MAX_S = 30                             # al analizar un archivo, reconoce con los primeros 30 s


def frame_spectrum(x):
    """Espectro (dB) de las últimas NFFT muestras; alimenta el espectrograma en vivo."""
    x = np.asarray(x[-NFFT:], dtype=float)
    if len(x) < NFFT:
        x = np.pad(x, (NFFT - len(x), 0))
    w = np.hanning(NFFT)
    X = np.fft.rfft(x * w)
    return 20 * np.log10(2 * np.abs(X) / w.sum() + 1e-9)


def static_spectrogram(x8):
    """Espectrograma (dB) de una señal completa."""
    _, t, S = spectrogram(x8, fs=fc.SR, window="hann", nperseg=NFFT,
                          noverlap=NFFT // 2, scaling="spectrum")
    return t, 10 * np.log10(S + 1e-12) + 3


class App:
    def __init__(self, root):
        self.root = root
        root.title("Reconocedor de canciones por huella acústica")
        root.geometry("1180x780")
        self.stream = None
        self.listening = False
        self.after_id = None
        self.job_q = None
        self.chunks = []
        self.tail = np.zeros(0)
        self.best = (None, 0)
        self.ref_id = 0                     # identifica la carga del espectrograma de referencia
        self.ref_q = None
        self.spec = np.full((len(FREQS), COLS), float(DB_MIN))
        self.build_ui()
        self.reset_live_view()
        self.refresh_songs()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Interfaz ----------------
    def build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="🎤 Escuchar", command=self.start_listen).pack(side="left", padx=2)
        ttk.Button(top, text="■ Detener", command=self.stop_listen).pack(side="left", padx=2)
        ttk.Label(top, text="  Máx. segundos:").pack(side="left")
        self.max_s = tk.IntVar(value=30)
        ttk.Spinbox(top, from_=10, to=120, width=5, textvariable=self.max_s).pack(side="left")
        ttk.Button(top, text="📂 Analizar archivo", command=self.analyze_file).pack(side="left", padx=(16, 2))
        ttk.Button(top, text="➕ Agregar canciones", command=self.add_songs).pack(side="left", padx=2)

        self.status = tk.StringVar(value="Listo.")
        ttk.Label(self.root, textvariable=self.status).pack(side="bottom", fill="x", padx=10, pady=4)

        self.result = tk.StringVar(value="—")
        ttk.Label(self.root, textvariable=self.result,
                  font=("Segoe UI", 18, "bold")).pack(fill="x", padx=10, pady=2)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        fig = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.ax_t = fig.add_subplot(3, 1, 1)
        self.ax_r = fig.add_subplot(3, 1, 2)
        self.ax_s = fig.add_subplot(3, 1, 3)

        # 1) Señal en el tiempo
        t = np.arange(int(FRAME_S * fc.SR)) / fc.SR
        self.line_t, = self.ax_t.plot(t, np.zeros_like(t), lw=0.8)
        self.ax_t.set_ylabel("Amplitud")

        # 2) (el espectrograma de la canción reconocida se dibuja en self.ax_r)

        # 3) Espectrograma de lo que escucha el micrófono
        self.im = self.ax_s.imshow(self.spec, origin="lower", aspect="auto", cmap="magma",
                                   vmin=DB_MIN, vmax=DB_MAX,
                                   extent=(-LIVE_SPAN, 0, 0, fc.SR / 2))
        self.ax_s.set_ylabel("Frecuencia (Hz)")

        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        side = ttk.Frame(body)
        side.pack(side="right", fill="y", padx=(8, 0))
        ttk.Label(side, text="Canciones en la base").pack()
        self.listbox = tk.Listbox(side, width=36)
        self.listbox.pack(fill="y", expand=True)

    def reset_live_view(self):
        self.ax_t.set_title(f"Señal de entrada (muestreada a {fc.SR} Hz, últimos {FRAME_S} s)")
        self.ax_t.set_xlabel("Tiempo (s)")
        self.ax_s.set_title(f"Espectrograma de lo que escucha el micrófono (últimos {LIVE_SPAN:.0f} s)")
        self.ax_s.set_xlabel("Tiempo (s)")
        self.spec[:] = DB_MIN
        self.im.set_data(self.spec)
        self.im.set_extent((-LIVE_SPAN, 0, 0, fc.SR / 2))
        self.ax_s.set_xlim(-LIVE_SPAN, 0)
        self.ax_s.set_ylim(0, fc.SR / 2)
        self.clear_ref()
        self.canvas.draw_idle()

    # ---------------- Espectrograma de la canción reconocida ----------------
    def clear_ref(self, msg="Aquí aparecerá el espectrograma de la canción reconocida"):
        self.ref_id += 1                        # descarta cargas pendientes
        self.ref_q = None
        self.ax_r.clear()
        self.ax_r.set_title("Espectrograma de la canción reconocida")
        self.ax_r.set_xticks([])
        self.ax_r.set_yticks([])
        self.ax_r.text(0.5, 0.5, msg, ha="center", va="center",
                       transform=self.ax_r.transAxes, color="gray")

    def show_ref(self, name, pos, dur):
        """Carga el archivo de la canción reconocida y dibuja su espectrograma."""
        path = fc.get_song_path(name)
        if not path or not os.path.exists(path):
            self.clear_ref(f"No tengo el archivo de '{name}'.\nVuelve a agregarla con ➕ Agregar canciones.")
            self.canvas.draw_idle()
            return
        self.clear_ref("Cargando espectrograma...")
        self.canvas.draw_idle()
        rid = self.ref_id
        q = queue.Queue()
        self.ref_q = q

        def worker():
            try:
                ts, S_db = static_spectrogram(fc.load_audio(path))
                q.put((rid, ("ok", name, ts, S_db, pos, dur)))
            except Exception as e:
                q.put((rid, ("error", str(e))))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(200, self.poll_ref, q)

    def poll_ref(self, q):
        if q is not self.ref_q:                 # ya hay una carga más nueva
            return
        try:
            rid, payload = q.get_nowait()
        except queue.Empty:
            self.root.after(200, self.poll_ref, q)
            return
        self.ref_q = None
        if rid != self.ref_id:
            return
        if payload[0] == "error":
            self.clear_ref(f"No se pudo abrir el archivo:\n{payload[1]}")
        else:
            self.draw_ref(*payload[1:])
        self.canvas.draw_idle()

    def draw_ref(self, name, ts, S_db, pos, dur):
        self.ax_r.clear()
        vmax = float(S_db.max())
        self.ax_r.imshow(S_db, origin="lower", aspect="auto", cmap="magma",
                         vmin=vmax - 80, vmax=vmax, extent=(0, ts[-1], 0, fc.SR / 2))
        self.ax_r.axvspan(pos, pos + dur, color="cyan", alpha=0.25, label="Lo que escuchaste")
        self.ax_r.set_xlim(0, ts[-1])
        self.ax_r.set_ylim(0, fc.SR / 2)
        self.ax_r.set_title(f"Espectrograma de la canción reconocida: {name}")
        self.ax_r.set_xlabel("Tiempo en la canción (s)")
        self.ax_r.set_ylabel("Frecuencia (Hz)")
        self.ax_r.legend(loc="upper right")

    def refresh_songs(self):
        self.listbox.delete(0, "end")
        try:
            for name, n in fc.list_songs():
                self.listbox.insert("end", f"{name}  ({n} huellas)")
        except Exception as e:
            self.status.set(f"No se pudo leer la base: {e}")

    # ---------------- Micrófono en vivo ----------------
    def start_listen(self):
        if self.listening:
            return
        if not fc.list_songs():
            messagebox.showinfo("Base vacía", "Primero agrega canciones con '➕ Agregar canciones'.")
            return
        try:
            import sounddevice as sd
            self.chunks = []
            self.tail = np.zeros(0)

            def callback(indata, frames, t, status):
                c = indata[:, 0].copy()
                self.chunks.append(c)
                self.tail = np.concatenate((self.tail, c))[-DEV_SR:]

            self.stream = sd.InputStream(samplerate=DEV_SR, channels=1,
                                         dtype="float32", callback=callback)
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Micrófono", f"No se pudo abrir el micrófono:\n{e}")
            return
        self.reset_live_view()
        self.listening = True
        self.best = (None, 0)
        self.t0 = time.time()
        self.last_try = 0.0
        self.result.set("Escuchando...")
        self.tick()

    def stop_listen(self):
        self.listening = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def tick(self):
        if not self.listening:
            return
        n = int(FRAME_S * DEV_SR)
        tail = self.tail
        if len(tail) >= n:
            seg8 = fc.resample(tail[-n:].astype(float), DEV_SR, fc.SR)
            self.draw_frame(seg8)
        el = time.time() - self.t0
        self.status.set(f"Escuchando... {el:.0f} s")

        if el >= LIVE_MIN_S and el - self.last_try >= RETRY_S:
            self.last_try = el
            audio = np.concatenate(self.chunks)[-WINDOW_S * DEV_SR:]
            x8 = fc.resample(audio.astype(float), DEV_SR, fc.SR)
            name, score, pos, ok = fc.recognize(x8)
            if ok:
                self.show_match(name, score, pos, len(x8) / fc.SR)
                self.stop_listen()
                return
            if name and score > self.best[1]:
                self.best = (name, score)
        if el >= self.max_s.get():
            self.stop_listen()
            self.result.set("No se reconoció ninguna canción")
            if self.best[0]:
                self.status.set(f"Mejor candidato: '{self.best[0]}' ({self.best[1]} votos), pero no es "
                                f"lo bastante claro. Acerca el micrófono o sube el volumen.")
            else:
                self.status.set("Sin coincidencias. Acerca el micrófono o sube el volumen.")
            return
        self.after_id = self.root.after(TICK_MS, self.tick)

    def show_match(self, name, score, pos, dur):
        self.result.set(f"🎵 {name}")
        self.status.set(f"{score} votos alineados · tu fragmento empieza aprox. en {pos:.0f} s de la canción")
        self.show_ref(name, pos, dur)

    def draw_frame(self, seg8):
        t = np.arange(len(seg8)) / fc.SR
        self.line_t.set_data(t, seg8)
        self.ax_t.set_xlim(0, t[-1])
        peak = max(0.02, float(np.abs(seg8).max()) * 1.2)
        self.ax_t.set_ylim(-peak, peak)
        self.spec = np.roll(self.spec, -1, axis=1)
        self.spec[:, -1] = frame_spectrum(seg8)
        self.im.set_data(self.spec)
        self.canvas.draw_idle()

    # ---------------- Archivo ----------------
    def analyze_file(self):
        path = filedialog.askopenfilename(
            title="Elige un audio",
            filetypes=[("Audio", "*.wav *.flac *.mp3 *.ogg"), ("Todos", "*.*")])
        if not path:
            return
        self.stop_listen()
        if not fc.list_songs():
            messagebox.showinfo("Base vacía", "Primero agrega canciones con '➕ Agregar canciones'.")
            return
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float64", always_2d=True)
        except Exception as e:
            messagebox.showerror("Archivo", f"No se pudo abrir el archivo:\n{e}")
            return
        x8 = fc.resample(data.mean(axis=1), sr, fc.SR)
        self.result.set("Analizando...")
        self.status.set(os.path.basename(path))
        self.clear_ref()
        self.root.update_idletasks()
        self.draw_static(x8)
        clip = x8[:FILE_MAX_S * fc.SR]
        name, score, pos, ok = fc.recognize(clip)
        if ok:
            self.show_match(name, score, pos, len(clip) / fc.SR)
        elif name:
            self.result.set("No se reconoció ninguna canción")
            self.status.set(f"Mejor candidato: '{name}' ({score} votos), pero no es lo bastante claro.")
        else:
            self.result.set("No se reconoció ninguna canción")
            self.status.set("Sin coincidencias.")

    def draw_static(self, x8):
        step = max(1, len(x8) // 4000)
        n = len(x8) // step * step
        blocks = x8[:n].reshape(-1, step)
        peaks = blocks[np.arange(len(blocks)), np.abs(blocks).argmax(axis=1)]
        t = np.arange(len(blocks)) * step / fc.SR
        self.line_t.set_data(t, peaks)
        self.ax_t.set_xlim(0, len(x8) / fc.SR)
        ymax = max(0.02, float(np.abs(x8).max()) * 1.1)
        self.ax_t.set_ylim(-ymax, ymax)
        self.ax_t.set_title(f"Señal del archivo ({len(x8) / fc.SR:.1f} s, remuestreada a {fc.SR} Hz)")
        self.ax_t.set_xlabel("Tiempo (s)")

        ts, S_db = static_spectrogram(x8)
        self.im.set_data(S_db)
        self.im.set_extent((0, ts[-1], 0, fc.SR / 2))
        self.ax_s.set_xlim(0, ts[-1])
        self.ax_s.set_ylim(0, fc.SR / 2)
        self.ax_s.set_title("Espectrograma del archivo analizado")
        self.ax_s.set_xlabel("Tiempo (s)")
        self.canvas.draw_idle()

    # ---------------- Agregar canciones a la base ----------------
    def add_songs(self):
        if self.job_q is not None:
            return
        paths = filedialog.askopenfilenames(
            title="Canciones a agregar a la base",
            filetypes=[("Audio", "*.wav *.flac *.mp3 *.ogg"), ("Todos", "*.*")])
        if not paths:
            return
        self.stop_listen()
        self.job_q = queue.Queue()
        q = self.job_q

        def worker():
            for p in paths:
                q.put(f"Procesando '{os.path.basename(p)}'...")
                try:
                    q.put(fc.add_song(p))
                except Exception as e:
                    q.put(f"[error] {os.path.basename(p)}: {e}")
            q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        self.poll_job()

    def poll_job(self):
        try:
            while True:
                msg = self.job_q.get_nowait()
                if msg is None:
                    self.job_q = None
                    self.refresh_songs()
                    self.status.set("Listo. Canciones agregadas a la base.")
                    return
                self.status.set(msg)
        except queue.Empty:
            pass
        self.root.after(200, self.poll_job)

    def on_close(self):
        self.stop_listen()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()