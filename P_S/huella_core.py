#!/usr/bin/env python3
"""
Reconocimiento de canciones por huella acústica (fingerprinting), estilo Shazam.
No modifica ni copia tus canciones: solo guarda sus huellas en huellas.db.

Instalación:  pip install numpy scipy soundfile sounddevice

Uso por terminal:
  python huella_core.py add "cancion.mp3" ["otra.wav" ...]   # agrega canciones a la base
  python huella_core.py addfolder ./musica                   # agrega una carpeta completa
  python huella_core.py listen 10                            # escucha 10 s del micrófono
  python huella_core.py file fragmento.wav                   # reconoce un archivo
  python huella_core.py list                                 # lista las canciones de la base
"""
import os
import sys
import sqlite3
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np
from scipy.signal import spectrogram, resample_poly
from scipy.ndimage import maximum_filter

# ---------------- Parámetros ----------------
SR = 8000                       # frecuencia de trabajo (cubre hasta 4 kHz)
NPERSEG = 1024                  # ventana FFT (128 ms)
NOVERLAP = 512                  # 50 % de solape -> 1 frame = 64 ms
PEAK_NEIGHBORHOOD = (20, 20)    # (bins de frecuencia, frames de tiempo)
MIN_AMPLITUDE_DB = -60          # ignora picos más débiles (relativo al máximo)
FAN_OUT = 10                    # pares por cada pico ancla
DT_MIN, DT_MAX = 1, 100         # ventana de emparejamiento en frames (< 128)
MIN_MATCHES = 10                # votos mínimos para aceptar una coincidencia
MARGIN = 2.0                    # el ganador debe tener >= MARGIN x los votos del siguiente mejor
HOP_S = (NPERSEG - NOVERLAP) / SR
# La base se guarda junto a este archivo, sin importar desde dónde ejecutes
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huellas.db")


# ---------------- Audio ----------------
def resample(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    f = Fraction(sr_to, int(sr_from)).limit_denominator(1000)
    return resample_poly(x, f.numerator, f.denominator)


def load_audio(path):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), sr, SR).astype(np.float32)


def record_mic(seconds, device_sr=44100):
    import sounddevice as sd
    print(f"Escuchando {seconds} s... pon la música cerca del micrófono.")
    rec = sd.rec(int(seconds * device_sr), samplerate=device_sr,
                 channels=1, dtype="float32")
    sd.wait()
    return resample(rec[:, 0].astype(np.float64), device_sr, SR).astype(np.float32)


# ---------------- Huellas ----------------
def find_peaks(signal):
    """Devuelve lista de (frame, bin_frecuencia) con los máximos locales."""
    if len(signal) < 2 * NPERSEG:
        return []
    _, _, S = spectrogram(signal, fs=SR, nperseg=NPERSEG,
                          noverlap=NOVERLAP, window="hann")
    S = 10 * np.log10(S + 1e-10)
    S -= S.max()
    local_max = maximum_filter(S, size=PEAK_NEIGHBORHOOD) == S
    mask = local_max & (S > MIN_AMPLITUDE_DB)
    freqs, times = np.nonzero(mask)
    return sorted(zip(times.tolist(), freqs.tolist()))


def make_pairs(peaks):
    """Empareja cada pico ancla con los siguientes -> [(t1, f1, t2, f2)]."""
    out = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        pares = 0
        for j in range(i + 1, n):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < DT_MIN:
                continue
            if dt > DT_MAX or pares >= FAN_OUT:
                break
            out.append((t1, f1, t2, f2))
            pares += 1
    return out


def make_hashes(peaks):
    """Convierte los pares de picos en hashes -> [(hash, tiempo_ancla)]."""
    return [((f1 << 17) | (f2 << 7) | (t2 - t1), t1)
            for t1, f1, t2, f2 in make_pairs(peaks)]


def fingerprint(signal):
    return make_hashes(find_peaks(signal))


# ---------------- Base de datos ----------------
def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            path TEXT
        );
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash INTEGER,
            song_id INTEGER,
            t INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash);
    """)
    # Bases creadas con versiones anteriores no tienen la columna "path"
    cols = [r[1] for r in con.execute("PRAGMA table_info(songs)")]
    if "path" not in cols:
        con.execute("ALTER TABLE songs ADD COLUMN path TEXT")
        con.commit()
    return con


def add_song(path):
    """Agrega una canción a la base. Devuelve un mensaje de texto."""
    name = os.path.splitext(os.path.basename(path))[0]
    full = os.path.abspath(path)
    con = db_connect()
    try:
        row = con.execute("SELECT path FROM songs WHERE name=?", (name,)).fetchone()
        if row:
            if not row[0]:      # canción agregada antes sin ruta: solo la guardamos
                con.execute("UPDATE songs SET path=? WHERE name=?", (full, name))
                con.commit()
                return f"'{name}' ya estaba en la base; se guardó su archivo para mostrar su espectrograma."
            return f"[omitida] '{name}' ya está en la base."
        hashes = fingerprint(load_audio(path))
        if not hashes:
            return f"[error] '{name}': no se pudieron extraer huellas (¿audio muy corto o vacío?)."
        cur = con.execute("INSERT INTO songs(name, path) VALUES (?, ?)", (name, full))
        sid = cur.lastrowid
        con.executemany("INSERT INTO fingerprints VALUES (?,?,?)",
                        [(h, sid, t) for h, t in hashes])
        con.commit()
        return f"Agregada '{name}' ({len(hashes)} huellas)."
    finally:
        con.close()


def get_song_path(name):
    """Ruta del archivo original de una canción de la base (o None)."""
    con = db_connect()
    try:
        row = con.execute("SELECT path FROM songs WHERE name=?", (name,)).fetchone()
    finally:
        con.close()
    return row[0] if row and row[0] else None


def add_folder(folder):
    exts = (".wav", ".mp3", ".flac", ".ogg")
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(exts):
            try:
                print(add_song(os.path.join(folder, f)))
            except Exception as e:
                print(f"[error] {f}: {e}")


def list_songs():
    """Devuelve [(nombre, cantidad_de_huellas), ...]."""
    con = db_connect()
    rows = con.execute("""
        SELECT s.name, COUNT(f.hash) FROM songs s
        LEFT JOIN fingerprints f ON f.song_id = s.id GROUP BY s.id ORDER BY s.id
    """).fetchall()
    con.close()
    return rows


# ---------------- Reconocimiento ----------------
def recognize(signal):
    """
    Devuelve (nombre, votos, posicion_s, ok).
    ok=True si la coincidencia es fiable. Si no, nombre/votos son del mejor candidato
    (nombre=None si no hubo ninguna coincidencia).
    """
    sample_hashes = fingerprint(signal)
    if not sample_hashes:
        return None, 0, 0.0, False

    by_hash = defaultdict(list)
    for h, t in sample_hashes:
        by_hash[h].append(t)

    con = db_connect()
    try:
        keys = list(by_hash.keys())
        votes = Counter()                      # (song_id, desfase) -> cuenta
        CHUNK = 500                            # evita el límite de variables de SQLite
        for i in range(0, len(keys), CHUNK):
            chunk = keys[i:i + CHUNK]
            q = ",".join("?" * len(chunk))
            rows = con.execute(
                f"SELECT hash, song_id, t FROM fingerprints WHERE hash IN ({q})",
                chunk).fetchall()
            for h, sid, t_db in rows:
                for t_s in by_hash[h]:
                    votes[(sid, t_db - t_s)] += 1
        if not votes:
            return None, 0, 0.0, False
        (sid, offset), score = votes.most_common(1)[0]
        # Mejor "rival": otra canción u otro desfase lejano de la misma canción
        runner = max((c for (s, o), c in votes.items()
                      if not (s == sid and abs(o - offset) <= 2)), default=0)
        name = con.execute("SELECT name FROM songs WHERE id=?", (sid,)).fetchone()[0]
    finally:
        con.close()
    ok = score >= MIN_MATCHES and score >= MARGIN * runner
    return name, score, max(offset, 0) * HOP_S, ok


def report(signal):
    name, score, pos, ok = recognize(signal)
    if ok:
        print(f"\n🎵 Canción: {name}")
        print(f"   Votos alineados: {score}")
        print(f"   Tu fragmento empieza aprox. en {pos:.1f} s de la canción")
    elif name:
        print(f"\nSin coincidencia clara (mejor candidato: '{name}', {score} votos).")
    else:
        print("\nSin coincidencias.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "add":
        for p in sys.argv[2:]:
            print(add_song(p))
    elif cmd == "addfolder":
        add_folder(sys.argv[2])
    elif cmd == "listen":
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10
        report(record_mic(secs))
    elif cmd == "file":
        report(load_audio(sys.argv[2]))
    elif cmd == "list":
        rows = list_songs()
        if not rows:
            print("La base está vacía.")
        for name, n in rows:
            print(f"{name}  ({n} huellas)")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()