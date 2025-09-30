# Geoguessr AI

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/geoguessr-ai/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/geoguessr-ai)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/geoguessr-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/geoguessrai_logo.png" width="50%" alt="Geoguessr AI Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [geoguessr-ai](#geoguessr-ai)
  - [Description](#description)
  - [🛠️ Prerequisites](#%EF%B8%8F-prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [📖 Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description

<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->

## 🛠️ Prerequisites

<!-- TODO: In this section you put what is needed for the program to run.
For example: OS version, programs, libraries, etc.  

-->

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

<!-- TODO: In this Section you describe how to install this project in its intended environment.(i.e. how to get it to run)  
-->

1. **Clone the repository**:

   ```sh
   git clone https://github.com/CogitoNTNU/geoguessr-ai.git
   cd geoguessr-ai
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

<!--
1. **Configure environment variables**:
    This project uses environment variables for configuration. Copy the example environment file to create your own:
    ```sh
    cp .env.example .env
    ```
    Then edit the `.env` file to include your specific configuration settings.
-->

1. **Set up pre commit** (only for development):
   ```sh
   uv run pre-commit install
   ```

## Usage

To run the project, run the following command from the root directory of the project:

```bash

```

<!-- TODO: Instructions on how to run the project and use its features. -->

### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the lastes commit on main by viewing the `gh-pages` branch on GitHub: [https://cogitontnu.github.io/geoguessr-ai/](https://cogitontnu.github.io/geoguessr-ai/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.

<table align="center">
    <tr>
        <!--
        <td align="center">
            <a href="https://github.com/NAME_OF_MEMBER">
              <img src="https://github.com/NAME_OF_MEMBER.png?size=100" width="100px;" alt="NAME OF MEMBER"/><br />
              <sub><b>NAME OF MEMBER</b></sub>
            </a>
        </td>
        -->
    </tr>
</table>

![Group picture](docs/img/team.png)

## Project structure
```bash
data/
  README.md                    # Data layout & sources
  benchmarks/                  # Public benchmarks (manifests/configs)

storage/                       #Database related stuff
  __init__.py
  s3_client.py                 # creates boto3 client (AWS or Brage/ECS)
  s3_images.py                 # upload/download image bytes, presigned URLs, metadata
  parquet_io.py                # upload/download parquet files (and list by prefix)

dataset_creation/              # uses storage/parquet_io.py to write/read manifests
  data_collection/
    collectors/
      streetview.py            # talks to provider APIs/SDKs
      local_ingest.py          # import from disks/zips
    pipelines/
      sv_grid_sweep.py         # e.g., sweep lat/lon grid, fetch headings
      sv_place_ids.py          # fetch by place IDs/list
    schemas.py                 # record schema (id, lat/lon, ts, source, license…)
    validate.py                # sanity checks (CRS, bounds, image dims)
    dedupe.py                  # SHA256/near-dup detection
    rate_limit.py              # backoff, retries
    utils.py
  geocell/
    h3_indexer.py              # Or s2/naive bins; compute geocells
    __init__.py
  builders/
    pretrain_builder.py        # Build pretrain parquet/manifests
    finetune_builder.py        # Build finetune datasets
    eval_builder.py            # Build eval/benchmark sets
  README.md
  __init__.py

preprocessing/
  embed.py                     # Feature extraction (e.g., CLIP)
  geo_augmentor.py             # Geo-specific augmentations
  dataset_preprocessing.py     # Image/label transforms
  geo_utils.py                 # CRS, distance, bounding-box helpers
  utils.py
  README.md
  __init__.py

models/
  encoder.py                   # Backbone or CLIP wrapper
  head.py                      # Prediction head
  layers/
    positional_encoder.py
    hedge.py
    __init__.py
  proto_refiner.py             # Optional refinement module
  utils.py
  __init__.py
  README.md

training/
  train_eval_loop.py           # Unified loop with checkpointing/logging
  train_modes.py               # Pretrain/finetune modes and configs
  __init__.py
  README.md

evaluation/
  metrics.py                   # Distance/top-k/cell metrics
  evaluate.py                  # Offline evaluation entrypoint
  __init__.py
  README.md

saved_models/                  # Checkpoints (git-ignored)
runs/                          # Logs (tensorboard/W&B) (git-ignored)

scripts/
  get_auxiliary_data.sh        # Download side data (coastlines, POIs, etc.)

config/
  settings.py                  # Central runtime config (paths, S3, hyperparams)

run.py                         # CLI to run preprocess/train/eval

```

### License

______________________________________________________________________

Distributed under the MIT License. See `LICENSE` for more information.
