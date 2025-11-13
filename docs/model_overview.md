## GeoGuessr-AI code overview

### 1) High-level by folder (concise)

- **backend**

  - `api.py` (unfinished): FastAPI app; basic routes for health, model/image lookup, upload to S3, dummy prediction.
  - `s3bucket.py`: S3 dataset I/O (ingest, manifests, snapshots, downloads, helpers).
  - `data.py`: Iterable PyTorch dataset that streams images from S3, returns image tensor + rich metadata.
  - `climate_preprocessing.py`: Sample Köppen–Geiger climate raster for points; map to coarse buckets.
  - `load_data_mod.py` (unfinished): Download most recent snapshot locally and attach file paths; dev/test helper.
  - `training.py` (unfinished): Stub training loop referencing a non-existent `ImageDataset`.

- **models**

  - `super_guessr.py` (partially unfinished): Main geocell-classification model on top of a vision encoder; panoramic handling, optional attention, top‑k cell outputs; references missing `smooth_labels`.
  - `layers/positional_encoder.py`: Sinusoidal positional encoder module.
  - `utils.py`: `ModelOutput` tuple, prediction and state-dict utilities.

- **preprocessing**

  - `dataset_preprocessing.py`: Build HF datasets; generate geocell labels; heading preprocessing; feature extraction; optional embedding path.
  - `geo_utils.py`: Haversine distance helpers; ECEF⇄LLA conversions; pairwise distance matrices.
  - `embed.py`: Multi-GPU embedding extraction utilities (saves .npy + indices).

- **pretrain**

  - `clip_embedder.py`: Frozen CLIP vision embedder wrapper; panorama-aware.
  - `pretrain_dataset.py`: Synthetic geo-caption dataset + CLIP pretraining entrypoint.
  - `utils.py`: Utility mirror (state-dict, predict).

- **training**

  - `train_eval_loop.py` (partially unfinished): Accelerator-based train/eval loop; references missing `ProtoRefiner` for refinement.
  - `train_modes.py`: Orchestrates CLIP pretraining and downstream finetuning on images or embeddings.

- **tests**

  - `test_clip.py`: Quick CLIP check on an image and text prompts.
  - `test_tinyvit.py`: TinyViT inference sanity check.
  - `run_sampling.py` (unfinished): Invokes external point-sampling pipeline not present in repo.

- **finetune_tinyvit**

  - `prepare_dataset.py`: Build local manifests with country labels from GADM; split train/val; download images.
  - `train_tinyvit_timm.py`: Minimal timm-based TinyViT trainer with metrics and checkpointing.
  - `extract_embeddings.py`: Load checkpoint and export embeddings to Parquet.
  - `README.md`: How-to for the above.

______________________________________________________________________

### 2) File-by-file method breakdown (status)

- **backend/api.py**

  - `lifespan(app)`: App startup/shutdown; ensure `/models` folder.
  - `Image`/`Model`: Pydantic models (unused fields in routes).
  - `read_root()`: Static HTML page.
  - `read_model(model_id)`: Returns dummy path. [unfinished]
  - `read_image(image_id)`: Returns fields from a default `Image()` (unpopulated). [unfinished]
  - `submit_image(file_path, s3_key)`: Upload single file or directory contents to S3 via `s3bucket.upload_image_to_s3`. (Relies on function name that does not exist as-is.) [unfinished]
  - `get_prediction(image_id)`: Returns dummy lat/lon. [unfinished]
  - `ping()`: Health endpoint.

- **backend/s3bucket.py**

  - Constants/clients: bucket, prefixes, boto3 client, retry-tuned.
  - `make_location_id(lat, lon, hex_len)`: Stable hash for location.
  - `img_key(location_id, heading)`: S3 key builder for image objects.
  - `put_json/get_json`: JSON I/O helpers to S3.
  - `upload_one_image(rec)`, `upload_batch(records, max_workers)`: Parallel image ingestion → DataFrame.
  - `write_batch_manifest(df, batch_date)`: Upload parquet manifest for batch.
  - `load_previous_snapshot()`, `load_latest_snapshot_df()`: Pointer-based snapshot loading.
  - `pick_image(rec, h)`: Nearest-heading image selector.
  - `merge_snapshot(prev, batch_df)`: Merge new batch into latest per (location_id, heading).
  - `write_new_snapshot(df_latest)`: Save snapshot parquet(s) + update pointer.
  - `parse_streetview_folder(root_dir)`, `records_from_streetview_index(idx, headings)`: Local folder → records for ingestion.
  - `list_keys(prefix)`, `read_parquet_prefix(prefix)`: S3 listing and read-concatenate.
  - `upload_dataset_from_folder(...)`: End-to-end ingestion pipeline; returns bookkeeping.
  - `download(...)`, `download_latest_images(...)`: Local cache of S3 images in parallel.
  - `load_points()`, `add_metadata()`, `get_snapshot_metadata()`: Convenience accessors.

- **backend/data.py**

  - `_ensure_s3_uri(path, bucket)`: Normalize to `s3://`.
  - `_build_fs(cache_dir, anon, s3_kwargs)`: fsspec S3 or filecache wrapper.
  - `GeoImageIterableDataset(IterableDataset)`: Streams images from S3; default transform to tensor; yields `(tensor, target_dict)`.
    - `__len__`, `_iter_shard()`, `_open_image(fs, uri)`, `__iter()`; robust retry and metadata packaging.

- **backend/climate_preprocessing.py**

  - `CLIMATE_DICT`: code→(label, desc) map.
  - `kg_to_bucket(kg_str, lat)`: Map KG class + latitude to coarse bucket.
  - `sample_koppen(df, raster_path, legend_map)`: Read raster, sample climate per point, append `Climate` column.

- **backend/load_data_mod.py**

  - `add_file_paths_to_df(dest_dir, overwrite, load_images)`: Load latest snapshot, download first N images locally, attach `local_image_path` and optional PIL images. [unfinished: list initialization bug and test-mode stub]

- **backend/training.py**

  - `train_model(data, labels, num_epochs)`: Stub loop using non-existent `ImageDataset`. [unfinished]

- **models/super_guessr.py**

  - `SuperGuessr(nn.Module)`: Vision-backbone → geocell classifier; panorama support; optional MHA with positional enc.; top‑k geocell logits.
    - `_set_hidden_size()`: Infer hidden size/mode from base model.
    - `_freeze_params()`: Optionally freeze and/or load pretrained head; partial TODO for tiny‑vit.
    - `load_geocells(path)`: Load centroids, register as non-trainable parameter.
    - `_move_to_cuda(...)`: Device placement in eval.
    - `load_state(path)`: Copy compatible weights.
    - `_assert_requirements(...)`: Input assertions.
    - `_to_one_hot(tensor)`: Scalar index → one‑hot.
    - `forward(...)`: Encode, combine panorama views, classify to geocells, return preds/top‑k/embedding; optional label smoothing via `smooth_labels` (missing). [partially unfinished]
    - `__str__`: Pretty print.

- **models/layers/positional_encoder.py**

  - `PositionalEncoder`: Sinusoidal PE with dropout; `forward` adds fixed PE.

- **models/utils.py**

  - `ModelOutput`: Named tuple fields for losses/preds.
  - `predict(model, dataset)`: HF Trainer predict wrapper.
  - `load_state_dict(self, state_dict, embedder=False)`: Flexible loader.

- **preprocessing/dataset_preprocessing.py**

  - `change_labels_for_classification(dataset)`: Swap labels to cell indices/one‑hots.
  - `load_geocell_df(path)`: Read geocell CSV, parse WKT polygons, attach geometry.
  - `generate_cell_labels(point, geocell_df, one_hot)`: Point→geocell index/one‑hot (covered_by; nearest fallback).
  - `preprocess_heading(example)`: Encode heading as 4×[sin,cos].
  - `generate_cell_labels_vector(labels, polygons)`: Vectorized labels to geocell indices.
  - `generate_label_cells(example, geocell_df, one_hot)`: Produce `labels` and `labels_clf` for each row.
  - `generate_label_mt(example)`: Multi‑task target vector from auxiliaries.
  - `extract_features(example, feature_extractor)`: HF pixel extraction; panorama stacking.
  - `get_embeddings/add_embeddings/__find_index`: Embedding utilities for precomputed embeddings.
  - `preprocess(dataset, geocell_path, embedder=None)`: Full pipeline; feature extraction or embedding path; geocell label creation. [uses many CPU workers]

- **preprocessing/geo_utils.py**

  - `haversine_np/haversine`: Pairwise distance (km) for numpy/torch.
  - `haversine_matrix(_np)`: All‑pairs distance matrices.
  - `lla2ecef(_np)`, `ecef2lla(_np)`, `cylindrical2geodetic`: Coordinate transforms.

- **preprocessing/embed.py**

  - `compute_embeddings(name, model, data, accelerator)`: Distributed feature extraction; save `.npy` + indices.
  - `embed_images(loaded_model, dataset)`: Prepare loaders with `EmbedDataset`, run on train/val/test.

- **pretrain/clip_embedder.py**

  - `CLIPEmbedding`: Wrap CLIP vision model; `_get_embedding`, `_pre_embed_hook`, `forward` (single or 4-view panorama).

- **pretrain/pretrain_dataset.py**

  - `PretrainDataset`: Image+synthetic caption dataset; random transforms; caption synthesis from metadata (country/region/town/climate/drive side/heading/month).
    - `_convert_to_row_index`, `_select_image`, `_is_valid`, `_select_caption`, `_random_transform`, `__getitem__`, `__len__`, `generate`, `accuracy`.
  - `collate_fn`: CLIP collator.
  - `pretrain(model, dataset, train_args, resume)`: CLIP pretraining harness with before/after accuracy. (Minimal; fine for experiments.)

- **pretrain/utils.py**

  - Same utilities as `models/utils.py`.

- **training/train_eval_loop.py**

  - `generate_profiler()`: Torch profiler config.
  - `evaluate_model(model, dataset, metrics, train_args, refiner, writer, step)`: Eval loop; optional `ProtoRefiner` refine step (class not present). [unfinished dependency]
  - `train_model(loaded_model, dataset, on_embeddings, train_args, metrics, patience, should_profile)`: Accelerator training loop; early stopping; saves best to `CURRENT_SAVE_PATH`.

- **training/train_modes.py**

  - `collate_fn`, `pretrain(...)`: CLIP pretraining wrapper.
  - `finetune_model(model, dataset, early_stopping, train_args)`: Load CLIP backbone, wrap `SuperGuessr`, finetune.
  - `finetune_on_embeddings(dataset, early_stopping, train_args)`: Train `SuperGuessr` directly on embeddings.

- **tests**

  - `test_clip.py`: Sanity CLI to rank prompts for an image via CLIP.
  - `test_tinyvit.py`: Sanity CLI for TinyViT ImageNet predictions.
  - `run_sampling.py`: Calls external sampling pipeline (`src.point_sampling_algorithm...`). [unfinished in this repo]

- **finetune_tinyvit**

  - `prepare_dataset.py`: Build labeled manifests from latest snapshot (country polygons), download images, split.
  - `train_tinyvit_timm.py`: timm TinyViT trainer (collate, loaders, training, metrics, checkpoint).
  - `extract_embeddings.py`: Load best checkpoint and export embeddings parquet.

______________________________________________________________________

### 3) Gaps to reach the paper architecture (what to implement, where)

- **Semantic geocell creation & hierarchy**

  - Add `preprocessing/geocell_creation.py`: build polygonal geocells (Voronoi or admin-aware), compute centroids, write CSV with WKT polygons and hierarchy levels. Provide OPTICS (or HDBSCAN) clustering over training points to seed cells and a coarse→fine hierarchy (4 granularities per paper).
  - Wire into `dataset_preprocessing.py` to consume this CSV instead of assuming preexisting `GEOCELL_PATH`.

- **Haversine label smoothing**

  - Add `preprocessing/label_smoothing.py` exporting `smooth_labels(dist_matrix, tau)` and re-export it from `preprocessing/__init__.py`.
  - Update imports in `models/super_guessr.py` to `from preprocessing.geo_utils import haversine_matrix` and `from preprocessing.label_smoothing import smooth_labels`.

- **Top‑K selection → refinement (within & across geocells)**

  - Implement `models/proto_refiner.py` (missing): given embedding, candidate cell indices/probs, refine to coordinates via prototype means, local kNN, or attention across candidates; optionally second-level refinement inside best cell clusters.
  - Integrate in `training/train_eval_loop.py` (currently references `ProtoRefiner`).

- **OPTICS-based cluster prototypes**

  - Add `preprocessing/cluster_prototypes.py`: precompute per‑geocell cluster centroids (OPTICS/DBSCAN), store as tensors for the refiner.

- **Panorama hierarchical attention (multi-view)**

  - Current `SuperGuessr` supports a simple self-attention path; finalize hyperparameters and heading encodings; optionally add cross‑view attention module for within‑geocell refinement.

- **Backend inference API**

  - Add `backend/inference.py`: load trained `SuperGuessr` + `ProtoRefiner`, image preprocessing, S3/local input, return lat/lon and top‑k geocells.
  - Update `backend/api.py` `/prediction` route to call the above (fix typo), return structured JSON.

- **Housekeeping**

  - Create `preprocessing/__init__.py` to expose `geo_utils.haversine_matrix` and new `label_smoothing.smooth_labels`.
  - Remove or finish `backend/training.py` and `backend/load_data_mod.py` or move to notebooks/scripts.

______________________________________________________________________

### 4) Prompt to generate a repository diagram

Copy this into your diagramming LLM and ask it to output a Mermaid diagram.

"""
You are a software diagram generator. Create a Mermaid diagram of the current repository modules and their relationships. Use subgraphs per folder. Draw arrows for imports/data flow. Style any NOT‑YET‑IMPLEMENTED files as gray with a dashed border. Also include suggested new files grouped in their folders with dashed borders and gray fill.

Requirements:

- Subgraphs: backend, models (and layers), preprocessing, pretrain, training, tests, finetune_tinyvit.
- Existing nodes:
  backend: api.py, s3bucket.py, data.py, climate_preprocessing.py, load_data_mod.py, training.py
  models: super_guessr.py, utils.py; layers/positional_encoder.py, layers/hedge.py
  preprocessing: dataset_preprocessing.py, geo_utils.py, embed.py
  pretrain: clip_embedder.py, pretrain_dataset.py, utils.py
  training: train_eval_loop.py, train_modes.py
  tests: test_clip.py, test_tinyvit.py, run_sampling.py
  finetune_tinyvit: prepare_dataset.py, train_tinyvit_timm.py, extract_embeddings.py
- Planned nodes (dashed + gray):
  preprocessing/geocell_creation.py, preprocessing/label_smoothing.py, preprocessing/cluster_prototypes.py,
  models/proto_refiner.py, backend/inference.py, preprocessing/__init__.py
- Key arrows (non-exhaustive):
  preprocessing.dataset_preprocessing -> models.super_guessr
  preprocessing.geo_utils -> models.super_guessr, models.layers.hedge
  pretrain.clip_embedder -> preprocessing.embed
  training.train_modes -> training.train_eval_loop, models.super_guessr
  training.train_eval_loop -> models.proto_refiner (dashed)
  backend.api -> backend.inference (dashed), backend.s3bucket
  finetune_tinyvit.prepare_dataset -> finetune_tinyvit.train_tinyvit_timm -> finetune_tinyvit.extract_embeddings
- Render with subgraphs and node styles:
  - Style not-implemented nodes: fill:#ddd,stroke-dasharray: 5 5
  - Style folder subgraphs with dashed borders.
    Output only Mermaid code inside a fenced block.
    """

## Diagram with state of current model implementatioon

![Current model state overview](../assets/current_model_state_overview.png)
