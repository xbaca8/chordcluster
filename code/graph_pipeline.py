"""
Reusable KNN-graph pipeline for the chord-transition similarity webpage.

This is the plot-free, importable counterpart of the logic in
`chord_transition_knn.ipynb`. It is the source of truth the web server uses to
(a) reproduce the per-mode feature tables that back the current graphs and
(b) add an external song to one of the graphs on demand.

Layout on disk (all paths relative to the repo root):
    code/songs_stochastic_mats.npz   -> cached analyze_all_songs() output
    gtzan/GTZAN_Enriched_V2.csv       -> song metadata (title/artist/genre/key)
    code/graph_data/{major,minor}.pkl -> persisted feature+metadata DataFrames
    code/uploads/                     -> audio files uploaded via the website
    knn_graph/graph_{major,minor}.json-> graph data consumed by knn_graph.html
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from song_analysis_utils import analyze_song

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR   = Path(__file__).resolve().parent
ROOT_DIR   = CODE_DIR.parent
NPZ_PATH   = CODE_DIR / "songs_stochastic_mats.npz"
CSV_PATH   = ROOT_DIR / "gtzan" / "GTZAN_Enriched_V2.csv"
TABLE_DIR  = CODE_DIR / "graph_data"
UPLOAD_DIR = CODE_DIR / "uploads"
GRAPH_DIR  = ROOT_DIR / "knn_graph"

# ── Graph construction constants (must match the notebook) ──────────────────────
K_NEIGHBORS = 3

CHORD_LABELS = {
    "Major": ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'],
    "Minor": ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII'],
}

# Non-feature (metadata) columns dropped before fitting the KNN model.
META_COLS = ['File Name', 'Genre', 'Title', 'Artist', 'GT Key', 'Key Name',
             'Mode', 'Predicted Key', 'Raw Chords', 'Romanized Chords']

# Genre tag assigned to songs uploaded through the website.
EXTERNAL_GENRE = "external"


# ── Feature helpers ─────────────────────────────────────────────────────────────
def feature_cols(mode: str) -> list[str]:
    """Ordered list of the 42 transition-feature column names for a mode."""
    labels = CHORD_LABELS[mode]
    return [f"{r} -> {c}" for r in labels for c in labels if r != c]


def matrix_to_features(matrix: np.ndarray, mode: str) -> dict[str, float]:
    """Flatten a song's 7x7 transition matrix into the named feature dict."""
    labels = CHORD_LABELS[mode]
    mat = np.asarray(matrix)[:7, :7]
    return {
        f"{labels[i]} -> {labels[j]}": float(mat[i, j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j
    }


def _add_transition_features(df: pd.DataFrame, mode: str, mats: dict) -> pd.DataFrame:
    """Attach the 42 transition columns to every row (matches the notebook)."""
    labels = CHORD_LABELS[mode]
    df = df.drop(columns=[c for c in df.columns if " -> " in c])

    cols = {name: [] for name in feature_cols(mode)}
    for _, song_row in df.iterrows():
        stem = os.path.splitext(song_row["File Name"])[0]
        mat = mats[mode][stem]["matrix"][:7, :7]
        for i, r in enumerate(labels):
            for j, c in enumerate(labels):
                if r != c:
                    cols[f"{r} -> {c}"].append(mat[i, j])
    return df.assign(**cols)


def _drop_all_zero_transitions(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Drop songs whose transition features are all zero (no usable transitions)."""
    cols = feature_cols(mode)
    keep = (df[cols] != 0).any(axis=1)
    return df[keep].reset_index(drop=True)


def has_usable_transitions(features: dict[str, float]) -> bool:
    return any(v != 0 for v in features.values())


# ── Base-table construction (reproduces the notebook) ───────────────────────────
def _load_stochastic_mats() -> dict:
    data = np.load(NPZ_PATH, allow_pickle=True)
    return data["stochastic_mats"][0]


def build_base_tables() -> dict[str, pd.DataFrame]:
    """Rebuild the per-mode feature tables from the cached matrices + CSV.

    Mirrors the notebook: merge metadata, drop duplicate (Title, Artist) pairs
    (keeping unknowns), split by mode, add transition features, drop all-zero
    songs. Persists the result to TABLE_DIR and returns it.
    """
    mats = _load_stochastic_mats()
    df = pd.read_csv(CSV_PATH)

    lookup = pd.DataFrame([
        {"stem": stem, "Mode": mode, "Predicted Key": info["key"],
         "Raw Chords": info["raw_chords"], "Romanized Chords": info["chords"]}
        for mode, songs in mats.items()
        for stem, info in songs.items()
    ])

    df["stem"] = df["File Name"].apply(lambda f: os.path.splitext(f)[0])
    df = df.merge(lookup, on="stem", how="left").drop(columns="stem")

    # Remove duplicate files with same Title & Artist (keep unknown title/artist).
    title_unknown  = df["Title"].isna()  | df["Title"].astype("string").str.strip().str.lower().eq("unknown title")
    artist_unknown = df["Artist"].isna() | df["Artist"].astype("string").str.strip().str.lower().eq("unknown artist")
    identifiable   = ~(title_unknown | artist_unknown)
    df = df[~(identifiable & df.duplicated(subset=["Title", "Artist"], keep="first"))].reset_index(drop=True)

    tables = {}
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("Major", "Minor"):
        sub = df[df["Mode"] == mode].reset_index(drop=True)
        sub = _add_transition_features(sub, mode, mats)
        sub = _drop_all_zero_transitions(sub, mode)
        tables[mode] = sub
        sub.to_pickle(TABLE_DIR / f"{mode.lower()}.pkl")
    return tables


def load_tables() -> dict[str, pd.DataFrame]:
    """Load persisted per-mode tables, building them on first use."""
    paths = {m: TABLE_DIR / f"{m.lower()}.pkl" for m in ("Major", "Minor")}
    if not all(p.exists() for p in paths.values()):
        return build_base_tables()
    return {m: pd.read_pickle(p) for m, p in paths.items()}


# ── KNN + graph export (matches the notebook's fixed build_knn_graph) ────────────
def knn_distances_indices(df_numerical: pd.DataFrame, k: int):
    """Mahalanobis KNN with self guaranteed at column 0.

    Identical-feature songs tie at distance 0 and sklearn's tie order is
    arbitrary, so we explicitly remove the query's own row wherever it lands and
    keep the k nearest non-self neighbors (column 0 stays self by convention).
    """
    model = NearestNeighbors(
        metric="mahalanobis",
        metric_params={"VI": np.cov(df_numerical, rowvar=False)},
        algorithm="auto",
    )
    model.fit(df_numerical.values)

    n = len(df_numerical)
    n_query = min(k + 1, n)
    raw_dist, raw_idx = model.kneighbors(df_numerical.values, n_neighbors=n_query)

    distances = np.zeros((n, k + 1))
    indices   = np.zeros((n, k + 1), dtype=int)
    for i in range(n):
        mask = raw_idx[i] != i
        nbr_idx  = raw_idx[i][mask][:k]
        nbr_dist = raw_dist[i][mask][:k]
        indices[i, 0] = i
        indices[i, 1:1 + len(nbr_idx)]    = nbr_idx
        distances[i, 1:1 + len(nbr_dist)] = nbr_dist
    return distances, indices


def export_graph_json(df: pd.DataFrame, distances, indices, out_path: Path) -> int:
    """Write nodes + edges + precomputed neighbors to JSON for knn_graph.html."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def summary(i):
        r = df.iloc[i]
        return {
            "file_name":        r["File Name"],
            "title":            r["Title"]         if isinstance(r["Title"], str)         else None,
            "artist":           r["Artist"]        if isinstance(r["Artist"], str)        else None,
            "genre":            r["Genre"],
            "key":              r["Predicted Key"] if isinstance(r["Predicted Key"], str) else None,
            "romanized_chords": r["Romanized Chords"] if isinstance(r["Romanized Chords"], list) else [],
        }

    nodes = []
    for i in range(len(df)):
        node = {"id": i, **summary(i)}
        node["neighbors"] = [
            {**summary(int(nbr)), "distance": float(d)}
            for nbr, d in zip(indices[i, 1:], distances[i, 1:])
        ]
        nodes.append(node)

    seen, edges = set(), []
    for i, (nbrs, dists) in enumerate(zip(indices, distances)):
        for nbr, d in zip(nbrs[1:], dists[1:]):
            j = int(nbr)
            if i == j or frozenset((i, j)) in seen:
                continue
            seen.add(frozenset((i, j)))
            edges.append({"from": i, "to": j, "weight": float(d)})

    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f)
    return len(nodes)


def rebuild_graph(mode: str, df: pd.DataFrame | None = None) -> int:
    """Rebuild and export the JSON graph for one mode. Returns node count."""
    if df is None:
        df = load_tables()[mode]
    df_numerical = df.drop(columns=[c for c in META_COLS if c in df.columns])
    distances, indices = knn_distances_indices(df_numerical, K_NEIGHBORS)
    out = GRAPH_DIR / f"graph_{mode.lower()}.json"
    return export_graph_json(df, distances, indices, out)


def rebuild_all_graphs() -> dict[str, int]:
    tables = load_tables()
    return {m: rebuild_graph(m, tables[m]) for m in tables}


# ── Add an external song ────────────────────────────────────────────────────────
def _safe_filename(name: str) -> str:
    """Sanitize an uploaded filename to a safe basename."""
    name = os.path.basename(name or "").strip() or "uploaded_song"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def add_song(audio_bytes: bytes, title: str, artist: str, filename: str,
             duration: int = 30) -> dict:
    """Analyze an uploaded song, add it to its mode's graph, and re-export JSON.

    Args:
        duration: seconds of audio to analyze (1–300, default 30).

    Returns a dict: {mode, file_name, title, artist, key, romanized_chords,
                     n_nodes}. Raises ValueError if the song yields no usable
    chord transitions.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(filename)
    wav_path = UPLOAD_DIR / safe
    wav_path.write_bytes(audio_bytes)

    # External files are not in the GTZAN lookup; force the key-estimator path.
    mode, key, raw, romanized, matrix = analyze_song(
        str(wav_path), duration=duration, ignore_GT_key=True
    )

    features = matrix_to_features(matrix, mode)
    if not has_usable_transitions(features):
        raise ValueError(
            "This song produced no usable chord transitions "
            "(all-zero feature vector), so it can't be added to the graph."
        )

    title  = (title or "").strip() or os.path.splitext(safe)[0]
    artist = (artist or "").strip() or "Unknown Artist"

    row = {
        "File Name": safe, "Genre": EXTERNAL_GENRE, "Title": title,
        "Artist": artist, "GT Key": -1, "Key Name": None, "Mode": mode,
        "Predicted Key": key, "Raw Chords": raw, "Romanized Chords": romanized,
        **features,
    }

    tables = load_tables()
    df = tables[mode]
    # Replace any prior upload with the same filename, then append.
    df = df[df["File Name"] != safe].reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame([row]).reindex(columns=df.columns)],
                   ignore_index=True)

    df.to_pickle(TABLE_DIR / f"{mode.lower()}.pkl")
    n_nodes = rebuild_graph(mode, df)

    return {
        "mode": mode.lower(), "file_name": safe, "title": title,
        "artist": artist, "key": key, "romanized_chords": romanized,
        "n_nodes": n_nodes,
    }


def delete_song(file_name: str, mode: str | None = None) -> dict:
    """Remove a previously uploaded ('external') song and rebuild its graph.

    Only songs tagged with EXTERNAL_GENRE may be deleted, so the API can never
    remove base GTZAN entries. `mode` ("major"/"minor") narrows the search when
    known; otherwise both tables are checked. Returns {mode, file_name, n_nodes}.
    Raises ValueError if the song is missing or not an uploaded song.
    """
    file_name = _safe_filename(file_name)
    tables = load_tables()

    if mode:
        wanted = mode.capitalize()
        search = [wanted] if wanted in tables else []
    else:
        search = list(tables.keys())

    for m in search:
        df = tables[m]
        hit = df["File Name"] == file_name
        if not hit.any():
            continue
        if not (df.loc[hit, "Genre"] == EXTERNAL_GENRE).all():
            raise ValueError(f"'{file_name}' is a base song and cannot be deleted.")

        df = df[~hit].reset_index(drop=True)
        df.to_pickle(TABLE_DIR / f"{m.lower()}.pkl")
        n_nodes = rebuild_graph(m, df)

        upload = UPLOAD_DIR / file_name
        if upload.exists():
            upload.unlink()

        return {"mode": m.lower(), "file_name": file_name, "n_nodes": n_nodes}

    raise ValueError(f"'{file_name}' not found in the {mode or 'graph'} data.")


if __name__ == "__main__":
    # Build base tables and export both graphs (sanity check / bootstrap).
    counts = rebuild_all_graphs()
    print("Rebuilt graphs:", counts)
