# sql_dataset_adapter.py
from __future__ import annotations
import sqlite3
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np
import torch


def _blob_to_tensor(blob: bytes, dim: int) -> torch.Tensor:
    """Decode float32 embedding blob with known last dim `dim`."""
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size % dim != 0:
        raise ValueError(f"Blob size {arr.size} not divisible by dim {dim}")
    shape = (arr.size // dim, dim) if (arr.size // dim) > 1 else (dim,)
    return torch.from_numpy(arr.reshape(shape))


class _SQLiteReader:
    """
    Minimal reader for a single SQLite file produced by your builders:
      CREATE TABLE samples (
        location_id TEXT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        heading INTEGER NOT NULL,
        capture_date TEXT,
        pano_id TEXT,
        batch_date TEXT,
        embedding BLOB NOT NULL,
        embedding_dim INTEGER NOT NULL,
        PRIMARY KEY (location_id, heading)
      ) WITHOUT ROWID;

    Lookup is by (location_id, heading).
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA mmap_size=268435456;")

    def fetch_many_loc_head(
        self, pairs: List[Tuple[str, int]]
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Returns a dict keyed by (location_id, heading) with:
          { "lat": float, "lon": float, "embedding": bytes, "embedding_dim": int }
        Missing keys just won't appear in the output dict.
        """
        if not pairs:
            return {}
        # Chunk IN clause under ~999 params. We have 2 params per row (loc_id, heading).
        out: Dict[Tuple[str, int], Dict[str, Any]] = {}
        chunk = 450  # 450*2=900 params < 999
        for i in range(0, len(pairs), chunk):
            sub = pairs[i : i + chunk]
            # Build compound OR list: (location_id=? AND heading=?)
            conds = ["(location_id = ? AND heading = ?)"] * len(sub)
            q = (
                "SELECT location_id, heading, lat, lon, embedding, embedding_dim "
                "FROM samples WHERE " + " OR ".join(conds)
            )
            params: List[Any] = []
            for loc, head in sub:
                params.extend([loc, int(head)])
            cur = self.conn.execute(q, params)
            for loc, head, lat, lon, emb, dim in cur.fetchall():
                out[(loc, int(head))] = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "embedding": emb,
                    "embedding_dim": int(dim),
                }
        return out


class DualSQLiteEmbeddingDataset:
    """
    Read-only adapter that merges CLIP + TinyViT embeddings from two SQLite files.

    __getitem__(indices) accepts either:
      - a list of (location_id:str, heading:int) tuples
      - OR a list of dicts with keys {"location_id":..., "heading":...}
      - OR a list of ints if/when you add a mapping from ids -> (location_id, heading)
        (you can pass an `id_to_lochead` callable for that)

    It returns a dict with:
      Locationid: List[str]
      heading:    List[int]
      photo:      List[Path|None]   # not stored in your DBs; kept as None placeholders
      emb_clip:   torch.Tensor      # (B, Dclip) or (B, 4, Dclip) if you stored panorama-avg OFF
      emb_tiny_vit: torch.Tensor    # (B, Dtiny) or (B, 4, Dtiny)
      labels:     torch.Tensor      # (B, 2) -> [lon, lat] for haversine
    """

    def __init__(
        self,
        clip_db_path: str,
        tinyvit_db_path: str,
        id_to_lochead: callable | None = None,
    ):
        self.clip = _SQLiteReader(clip_db_path)
        self.tiny = _SQLiteReader(tinyvit_db_path)
        self.id_to_lochead = (
            id_to_lochead  # Optional mapping: int -> (location_id, heading)
        )

    @staticmethod
    def _normalize_indices(indices: Iterable[Any]) -> List[Tuple[str, int]]:
        pairs: List[Tuple[str, int]] = []
        for x in indices:
            if isinstance(x, tuple) and len(x) == 2:
                loc, head = x
                pairs.append((str(loc), int(head)))
            elif isinstance(x, dict):
                pairs.append((str(x["location_id"]), int(x["heading"])))
            elif isinstance(x, (list,)):
                # support ["loc", 90]
                if len(x) != 2:
                    raise ValueError("List index must be [location_id, heading]")
                pairs.append((str(x[0]), int(x[1])))
            elif isinstance(x, (int, np.integer)):
                raise ValueError(
                    "Got integer id but no id_to_lochead mapping. "
                    "Pass id_to_lochead when constructing DualSQLiteEmbeddingDataset."
                )
            else:
                raise ValueError(f"Unsupported index format: {type(x)}")
        return pairs

    def __getitem__(self, indices: List[Any]) -> Dict[str, Any]:
        # Normalize to (location_id, heading) pairs
        try:
            pairs = self._normalize_indices(indices)
        except ValueError:
            if self.id_to_lochead is None:
                raise
            # Try mapping integer ids -> pairs
            mapped = [self.id_to_lochead(i) for i in indices]
            pairs = self._normalize_indices(mapped)

        # Fetch from both DBs
        clip_rows = self.clip.fetch_many_loc_head(pairs)
        tinv_rows = self.tiny.fetch_many_loc_head(pairs)

        locids, heads, photos = [], [], []
        emb_clip, emb_tiny, labels = [], [], []

        for key in pairs:
            loc, head = key
            c = clip_rows.get(key)
            t = tinv_rows.get(key)
            if c is None or t is None:
                # Skip missing rows gracefully
                continue

            locids.append(loc)
            heads.append(int(head))
            photos.append(None)  # your SQLite builders don't store image_path

            # Decode blobs
            clip_vec = _blob_to_tensor(c["embedding"], c["embedding_dim"])
            tiny_vec = _blob_to_tensor(t["embedding"], t["embedding_dim"])
            emb_clip.append(clip_vec)
            emb_tiny.append(tiny_vec)

            # Labels as [lon, lat] for haversine; consistent with your ProtoRefiner
            labels.append([float(c["lon"]), float(c["lat"])])

        out: Dict[str, Any] = {
            "Locationid": locids,
            "heading": heads,
            "photo": photos,  # List[Path|None]; kept as None
            "emb_clip": torch.stack(emb_clip, dim=0) if emb_clip else torch.empty(0),
            "emb_tiny_vit": torch.stack(emb_tiny, dim=0)
            if emb_tiny
            else torch.empty(0),
            "labels": torch.tensor(labels, dtype=torch.float32)
            if labels
            else torch.empty((0, 2)),
        }
        return out
