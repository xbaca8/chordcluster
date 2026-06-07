import glob
import os

import librosa
import numpy as np
import pandas as pd
import scipy.linalg
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import zscore
from typing import List
import matplotlib.pyplot as plt

from chord_extractor.extractors import Chordino

HOP_LENGTH = 512
MAX_TIME = 30

GTZAN_CSV = os.path.join(os.path.dirname(__file__), '..', 'gtzan', 'GTZAN_Enriched_V2.csv')

# GT Key index order from KeyEnumeration.txt (0-11 major, 12-23 minor)
# Converted to flat notation to match the rest of the codebase
_GT_PITCH_NAMES = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab']

_gtzan_lookup: dict | None = None  # filename -> gt_key int, loaded once


def _load_gtzan_lookup() -> dict:
    global _gtzan_lookup
    if _gtzan_lookup is None:
        df = pd.read_csv(GTZAN_CSV)
        _gtzan_lookup = dict(zip(df['File Name'], df['GT Key'].astype(int)))
    return _gtzan_lookup


def _gt_key_to_str(gt_key: int) -> str | None:
    """Convert a GT Key index (0-23) to a key string like 'C Major' / 'A Minor'.
    Returns None for -1 (unknown/modulation)."""
    if gt_key == -1:
        return None
    if 0 <= gt_key <= 11:
        return f"{_GT_PITCH_NAMES[gt_key]} Major"
    if 12 <= gt_key <= 23:
        return f"{_GT_PITCH_NAMES[gt_key - 12]} Minor"
    return None

# ── Key Estimator ─────────────────────────────────────────────────────────────
# Taken from: https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd

@dataclass
class KeyEstimator:
    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = np.asarray(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor = np.asarray(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    def __post_init__(self):
        self.major = zscore(self.major)
        self.major_norm = scipy.linalg.norm(self.major)
        self.major = scipy.linalg.circulant(self.major)

        self.minor = zscore(self.minor)
        self.minor_norm = scipy.linalg.norm(self.minor)
        self.minor = scipy.linalg.circulant(self.minor)

    def __call__(self, x: np.array) -> tuple[np.array, np.array]:
        x = zscore(x)
        x_norm = scipy.linalg.norm(x)
        coeffs_major = self.major.T.dot(x) / self.major_norm / x_norm
        coeffs_minor = self.minor.T.dot(x) / self.minor_norm / x_norm
        return coeffs_major, coeffs_minor


# ── Circle of Fifths ──────────────────────────────────────────────────────────

circle_of_fifths = {
    'Major': ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F'],
    'Minor': ['A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D'],
}


# ── Chord Romanization ────────────────────────────────────────────────────────

def get_romanized_chords(key: str, chords: List) -> List[str]:
    '''
    Takes a musical key in the form '[Note] [Mode]' where the Mode is
    either 'Major' or 'Minor', and a sequence of chords from Chordino, and
    then converts the chords to their respective roman numerals based on
    music theory rules.

    When considering the chords from Chordino, we only consider the chord
    root and not if it is major or minor -> we let the key determine that 
    because the Chordino chords seem to be less accurate than key.
    '''
    scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    tonic = key[:-6]  # Remove " Major" or " Minor"
    tonic_index = scale.index(tonic)
    rotated_chromatic_scale = scale[tonic_index:] + scale[:tonic_index]

    if key.endswith(" Major"):
        steps = [0, 2, 4, 5, 7, 9, 11]
        step_to_roman = {1: 'I', 2: 'ii', 3: 'iii', 4: 'IV', 5: 'V', 6: 'vi', 7: 'vii°'}
    elif key.endswith(" Minor"):
        steps = [0, 2, 3, 5, 7, 8, 10]
        step_to_roman = {1: 'i', 2: 'ii°', 3: 'III', 4: 'iv', 5: 'v', 6: 'VI', 7: 'VII'}

    rotated_scale = [rotated_chromatic_scale[i] for i in steps]

    romanized_sequence = []
    for item in chords:
        chord = item.chord
        if chord == 'N':
            continue
        root = chord[0]
        if len(chord) > 1 and chord[1] == 'b':
            root = chord[0:2]
        if root not in rotated_scale:
            romanized_sequence.append('*')
            continue
        root_ind = rotated_scale.index(root)
        romanized_sequence.append(step_to_roman[root_ind + 1])

    return romanized_sequence


def _chord_root(chord: str) -> str:
    """Extract the root note from a Chordino chord label. 'Dm7' → 'D', 'Dbmaj' → 'Db'."""
    return chord[:2] if len(chord) > 1 and chord[1] == 'b' else chord[:1]


def _is_minor_chord_label(chord: str) -> bool:
    """True if a Chordino chord label denotes a minor chord. 'Dm' → True, 'Dmaj7' → False."""
    suffix = chord[len(_chord_root(chord)):]
    return suffix.startswith('m') and not suffix.startswith('maj')


def _count_tonics(romanized: List[str], key: str) -> int:
    """Count tonic chords ('I' for Major, 'i' for Minor) in a romanized sequence."""
    return romanized.count('I' if key.endswith(' Major') else 'i')


def get_best_romanized_chords(key: str, chords: List) -> tuple[List[str], str]:
    """
    Choose the best key for `chords` from a small candidate set built around
    `key` (its fourth, fifth, relative, parallel), and return the romanized
    sequence for that key plus the key itself.

    Selection rules:
      0. Single-tonic short-circuit — if Chordino reports only one chord root,
         use it directly; mode comes from the first chord's quality ('m' → Minor).
      1. Fewest *effective* breaks (raw '*' count plus a switching cost) wins.
         The original (input/KS-predicted) key has cost 0 and is the prior;
         every other candidate pays a cost so over-correction is harder.
            fourth, fifth: +1   (need ≥2 fewer breaks, or 1 fewer + more tonics)
            relative:      +2   (need ≥3 fewer breaks)
            parallel:      +0   (usually a KS mode-detection slip; easy to flip)
      2. Tiebreaker: most tonic chords ('I' / 'i'), also considering the switching cost.
      3. Final tiebreaker (priority hierarchy):
            original (Major only) > fourth > fifth > relative > parallel > original (Minor)
         For a Minor input, this naturally prefers the parallel- or relative-Major
         candidate over the original Minor when they tie on effective breaks.
    """
    # ── Update 3: single-tonic short-circuit ─────────────────────────────────
    non_N = [c.chord for c in chords if c.chord != 'N']
    if non_N and len({_chord_root(c) for c in non_N}) == 1:
        single_mode = "Minor" if _is_minor_chord_label(non_N[0]) else "Major"
        single_key  = f"{_chord_root(non_N[0])} {single_mode}"
        print(f"Key selected: {single_key}  (single-tonic short-circuit)")
        return get_romanized_chords(single_key, chords), single_key

    # ── Build candidate keys ─────────────────────────────────────────────────
    tonic      = key.split()[0]
    mode       = "Major" if key.endswith(" Major") else "Minor"
    other_mode = "Minor" if mode == "Major" else "Major"

    cof = circle_of_fifths[mode]
    idx = cof.index(tonic)

    fifth_tonic    = cof[(idx + 1) % 12]
    fourth_tonic   = cof[(idx - 1) % 12]
    relative_tonic = circle_of_fifths[other_mode][idx]

    original_key = f"{tonic} {mode}"
    fourth_key   = f"{fourth_tonic} {mode}"
    fifth_key    = f"{fifth_tonic} {mode}"
    relative_key = f"{relative_tonic} {other_mode}"
    parallel_key = f"{tonic} {other_mode}"

    candidates = [original_key, fourth_key, fifth_key, relative_key, parallel_key]
    results = {k: get_romanized_chords(k, chords) for k in candidates}
    breaks  = {k: seq.count('*')         for k, seq in results.items()}
    tonics  = {k: _count_tonics(seq, k)  for k, seq in results.items()}

    # Cost added to `breaks` to make leaving the original key harder.
    switching_cost = {
        original_key: 0,
        fourth_key:   1,
        fifth_key:    1,
        relative_key: 2,
        parallel_key: 1,
    }

    # Lower priority value = preferred when (effective breaks, tonics) are tied.
    priority = {
        original_key: 0 if mode == "Major" else 5,
        fourth_key:   1,
        fifth_key:    2,
        relative_key: 3,
        parallel_key: 4,
    }
    best_key = min(
        candidates,
        key=lambda k: (breaks[k] + switching_cost[k], -tonics[k] + switching_cost[k], priority[k]),
    )

    print(f"Key selected: {best_key}  (breaks: {breaks}, tonics: {tonics})")
    return results[best_key], best_key


# ── Chord-Change Transition Matrix ───────────────────────────────────────────

def chord_change_matrix(key: str, romanized_chords: List) -> np.ndarray:
    if key.endswith(' Major'):
        roman_to_step = {'I': 1, 'ii': 2, 'iii': 3, 'IV': 4, 'V': 5, 'vi': 6, 'vii°': 7, '*': 8}
    elif key.endswith(' Minor'):
        roman_to_step = {'i': 1, 'ii°': 2, 'III': 3, 'iv': 4, 'v': 5, 'VI': 6, 'VII': 7, '*': 8}

    matrix = np.zeros((8, 8))
    for i in range(len(romanized_chords) - 1):
        curr = roman_to_step[romanized_chords[i]]
        nxt  = roman_to_step[romanized_chords[i + 1]]
        matrix[curr - 1, nxt - 1] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix /= row_sums
    return matrix

def plot_chromagram(y, sr, duration):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        chroma,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="chroma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax)
    ax.set_title(f"Chromagram (chroma_stft) — first {duration}s")
    plt.tight_layout()
    plt.show()

    return chroma

# ── Song Analysis Pipeline ───────────────────────────────────────────────────

def analyze_song(wav_file: str, duration = None, ignore_GT_key: bool = False) -> tuple[str, str, list, list, np.ndarray]:
    """
    Full pipeline:
      1. Load audio and apply HPSS
      2. Extract chords with Chordino
      3. Determine key:
         - If the filename is in GTZAN_Enriched_V2.csv and GT Key != -1, use that
           ground-truth (GT) key directly with get_romanized_chords.
         - Otherwise fall back to KeyEstimator + get_best_romanized_chords.
      4. Compute and return the mode, final key, and chord-change transition matrix

    Returns: (mode: str, final_key: str, matrix: np.ndarray)
    """
    sharp_to_flat = {'C#': 'Db', 'D#': 'Eb', 'E#': 'F', 'F#': 'Gb',
                     'G#': 'Ab', 'A#': 'Bb', 'B#': 'C'}

    def normalize_chord(chord_str):
        if len(chord_str) > 1 and chord_str[1] == '#':
            return sharp_to_flat[chord_str[:2]] + chord_str[2:]
        return chord_str

    y, sr = librosa.load(wav_file, sr=None, mono=True, duration=duration)
    y_harmonic, _ = librosa.effects.hpss(y)

    chordino   = Chordino(roll_on=1)
    raw_chords = chordino.extract(wav_file, duration=duration)
    chords     = [c._replace(chord=normalize_chord(c.chord)) for c in raw_chords]

    # --- Key determination ---
    filename  = os.path.basename(wav_file)
    lookup    = _load_gtzan_lookup()
    gt_key_str = _gt_key_to_str(lookup[filename]) if filename in lookup else None

    romanized: List[str] = []
    final_key: str | None = None
    if gt_key_str is not None and not ignore_GT_key:
        romanized = get_romanized_chords(gt_key_str, chords)
        # If the GT key romanizes to mostly breaks ('*'), it's a poor fit for the
        # detected chords -> fall back to generating the key from the audio.
        if romanized and romanized.count('*') > len(romanized) / 2:
            print(f"GT key: {gt_key_str}  (rejected: >half breaks, generating key)")
            final_key = None
        else:
            print(f"GT key: {gt_key_str}")
            final_key = gt_key_str

    if final_key is None:
        # Get chroma features
        chromagram  = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH)
        mean_chroma = np.mean(chromagram, axis=1)

        # Use KS algorithm to make an initial prediction of the key
        coeffs_major, coeffs_minor = KeyEstimator()(mean_chroma)
        pitch_classes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        all_coeffs    = np.concatenate([coeffs_major, coeffs_minor])
        best_idx      = np.argmax(all_coeffs)
        predicted_key = (f"{pitch_classes[best_idx % 12]} Major"
                         if best_idx < 12 else
                         f"{pitch_classes[best_idx % 12]} Minor")
        print(f"Predicted key: {predicted_key}")

        # With the initial key prediction and the chords, test other keys with the same
        # set of chords in
        romanized, final_key = get_best_romanized_chords(predicted_key, chords)

    raw = [c.chord for c in chords if c.chord != 'N']
    print(f"Chords (raw):   {raw}")
    print(f"Chord sequence: {romanized}")

    mode = "Major" if final_key.endswith(" Major") else "Minor"
    return mode, final_key, raw, romanized, chord_change_matrix(final_key, romanized)


def analyze_all_songs(root_dir: str, ignore_GT_key: bool = False) -> dict:
    """
    Run analyze_song on every .wav file under root_dir and store results in a
    nested dictionary keyed by mode, then by song title.

    Returns:
        {
            "Major": { "song_title": {"matrix": np.ndarray, "key": str,
                                      "raw_chords": list[str],
                                      "chords": list[str]}, ... },
            "Minor": { ... },
        }
    """
    wav_files = glob.glob(f"{root_dir}/**/*.wav", recursive=True)
    print(f"Found {len(wav_files)} .wav files\n")

    results = {"Major": {}, "Minor": {}}

    for i, wav_file in enumerate(wav_files):
        title = os.path.splitext(os.path.basename(wav_file))[0]
        print(f"[{i+1}/{len(wav_files)}] {title}")
        try:
            mode, key, raw, romanized, matrix = analyze_song(wav_file, duration=30, ignore_GT_key=ignore_GT_key)
            if len(romanized) < 3:
                print(f"  SKIPPED: only {len(romanized)} chord(s) detected.")
                continue
            results[mode][title] = {"matrix": matrix, "key": key,
                                    "raw_chords": raw, "chords": romanized}
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    return results


# ── Steady-State Distributions ───────────────────────────────────────────────

def get_steady_state(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the steady-state distribution of a row-stochastic Markov matrix.
    Finds the left eigenvector of P corresponding to eigenvalue 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady_state = eigenvectors[:, idx].real
    steady_state = np.abs(steady_state)
    return steady_state / steady_state.sum()


def steady_state_vec(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.T
    dim = matrix.shape[0]
    q = matrix - np.eye(dim)
    q = np.c_[q, np.ones(dim)]
    QTQ = np.dot(q, q.T)
    return np.linalg.solve(QTQ, np.ones(dim))


# ── PCCA+ Spectral Clustering ─────────────────────────────────────────────────

def _isa(X: np.ndarray, m: int) -> List[int]:
    """
    Inner Simplex Algorithm: find m rows of X as simplex vertices.
    Uses sequential Gram-Schmidt orthogonalisation to locate extremal points.
    """
    B = X.copy().astype(float)
    vertices = [int(np.argmax(np.linalg.norm(B, axis=1)))]
    for _ in range(1, m):
        v = B[vertices[-1]].copy()
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v /= norm
            B -= np.outer(B @ v, v)
        norms = np.linalg.norm(B, axis=1)
        for vi in vertices:
            norms[vi] = -np.inf
        vertices.append(int(np.argmax(norms)))
    return vertices


def pcca_plus(P: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    PCCA+ spectral clustering algorithm (arXiv:2206.14537).

    Finds m metastable clusters of the n states of a row-stochastic Markov
    chain by computing a fuzzy membership matrix χ = X·A, where X contains
    the m dominant right eigenvectors of P and A is optimised for crispness.

    Args:
        P : (n, n) row-stochastic transition matrix
        m : number of metastable clusters

    Returns:
        chi : (n, m) membership matrix – rows sum to 1, values in [0, 1]
        A   : (m, m) optimal transformation matrix
    """
    n = P.shape[0]
    assert 1 < m <= n, "m must satisfy 1 < m ≤ n"

    eigenvalues, eigenvectors = np.linalg.eig(P)
    order = np.argsort(eigenvalues.real)[::-1]
    X = eigenvectors[:, order[:m]].real  # (n, m)

    vertices = _isa(X, m)
    S  = X[vertices, :]
    A0 = np.linalg.inv(S)

    def neg_crispness(a_flat):
        A   = a_flat.reshape(m, m)
        chi = X @ A
        obj = -np.sum(chi ** 2)
        pen = 1e4 * (
            np.sum(np.maximum(0.0, -chi) ** 2) +
            np.sum(np.maximum(0.0,  chi - 1.0) ** 2) +
            np.sum((chi.sum(axis=1) - 1.0) ** 2)
        )
        return obj + pen

    result = minimize(
        neg_crispness, A0.flatten(), method="Nelder-Mead",
        options={"maxiter": 100_000, "xatol": 1e-10, "fatol": 1e-10, "adaptive": True},
    )
    A_opt = result.x.reshape(m, m)

    chi = X @ A_opt
    chi = np.clip(chi, 0.0, 1.0)
    chi /= chi.sum(axis=1, keepdims=True)

    return chi, A_opt
