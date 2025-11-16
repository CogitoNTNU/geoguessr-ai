#!/usr/bin/env python3
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import (
    Dataset,
    enable_progress_bar,
    disable_progress_bar,
)
from preprocessing.geo_utils import haversine
from models.utils import ProtoDataManager
import sys
from pretrain.clip_embedder import CLIPEmbedding
from pretrain.tinyvit_embedder import TinyViTEmbedding
from main_coordinator_idun_s3 import LocalGeoMapDataset
import numpy as np
from training.load_sqlite_dataset import load_sqlite_panorama_dataset

# Cluster refinement model

PROTO_PATH = "data/geocells/proto_df.csv"
DATASET_PATH = "data/datasets/embeddings_dataset"


class ProtoRefiner(nn.Module):
    """Proto-Net refinement model"""

    def __init__(
        self,
        topk: int = 5,
        max_refinement: int = 1000,
        temperature: float = 1.6,
        proto_path: str = PROTO_PATH,
        protos: str = None,
        verbose: bool = False,
        clip_db_path: str = "data/sqlite/clip/dataset.sqlite",  # correct?
        tinyvit_db_path: str = "data/sqlite/tinyvit/dataset.sqlite",  # correct?
    ):
        """Proto-Net refinement model

        Args:
            topk (int, optional): number geocell candidates to consider.
                Defaults to 5.
            max_refinement (int, optional): max refinement distance in km.
                Defaults to 1000.
            temperature (float, optional): temperature influencing the softmax strength of
                the refiner probabilities. Defaults to 1.6.
            proto_path (str, optional): path to proto-cluster refinement file.
                Defaults to PROTO_PATH.
            dataset_path (str, optional): path to file containing embeddings for training dataset.
                The embeddings must be the ones produced by the SuperGuessr model for which
                guesses are refined. Defaults to DATASET_PATH.
            protos (List[Dataset], optional): directly supplied protos if needed. Defaults to None.
            verbose (bool, optional): Whether to print out processes in detail.
                Defaults to False.
        """
        super(ProtoRefiner, self).__init__()

        # Variables
        self.topk = topk
        self.max_refinement = max_refinement
        self.verbose = verbose
        print("Initializing embedder...")
        self.embedder = Embeddings()

        # TODO: get the needed data from the dataset through sql querying
        # Check all the places where self.dataset is used
        # self.dataset = {
        #     "train": DualSQLiteEmbeddingDataset(clip_db_path, tinyvit_db_path)
        # }

        # Load prototypes
        self.proto_df = pd.read_csv(proto_path)
        self.proto_manager = ProtoDataManager(self.proto_df)

        # Load prototype index dataframe
        self.proto_df["geocell_index"] = self.proto_df["geocell_index"].astype(int)
        self.num_geocells = self.proto_df["geocell_index"].max() + 1
        # self.proto_df = self.proto_df.set_index("geocell_id") The geocell_id is not unique and the file contains all the clusters for every geocell

        # Generate prototypes for every geocells
        # TODO: See what we need to load here
        if protos is None:
            self.load_prototypes()
            self.protos: List[Dataset]

            if len(self.protos) != self.num_geocells:
                raise ValueError(
                    "Number of loaded prototypes does not match number of geocells."
                )
            elif len(self.protos) is None:
                raise ValueError("Prototypes could not be loaded.")

            for i in range(len(self.protos)):
                if self.protos[i] is None:
                    continue
                self.protos[i].save_to_disk(f"data/geocells/protos/proto_{i}")
        else:
            self.protos = [None] * self.num_geocells
            for i in range(self.num_geocells):
                try:
                    self.protos[i] = Dataset.load_from_disk(
                        f"data/geocells/protos/proto_{i}"
                    )
                except FileNotFoundError:
                    self.protos[i] = None
            if verbose:
                print("Loaded prototypes from disk.")

        # Learnable parameters
        self.temperature = Parameter(torch.tensor(temperature), requires_grad=False)
        self.geo_scaling = Parameter(torch.tensor(20.0), requires_grad=False)

    def __str__(self):
        rep = "ProtoRefiner(\n"
        rep += f"\ttopk\t\t= {self.topk}\n"
        rep += f"\tmax_refinement\t= {self.max_refinement}\n"
        rep += f"\ttemperature\t= {self.temperature.data.item()}\n"
        rep += f"\tgeo_scaling\t= {self.geo_scaling.data.item()}\n"
        rep += ")"
        return rep

    def forward(
        self,
        embedding: Tensor = None,
        initial_preds: Tensor = None,
        candidate_cells: Tensor = None,
        candidate_probs: Tensor = None,
    ):
        """Forward function for proto refinement model.

        Args:
            embedding (Tensor): CLIP embeddings of images.
            initial_preds (Tensor): initial predictions.
            candidate_cells (Tensor): tensor of candidate geocell predictions.
            candidate_probs (Tensor): tensor of probabilities assigned to
                each candidate geocell. Defaults to None.
        """
        assert self.topk <= candidate_cells.size(1), (
            '"topk" parameter must be smaller or equal to the number of geocell candidates \
             passed into the forward function.'
        )

        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        # If no probabilities are passed, only consider first cell candidate
        if candidate_probs is None:
            candidate_probs = torch.zeros_like(candidate_cells)
            candidate_probs[:, 0] = 1

        # Setup variables
        guess_index = []
        preds_LLH = []
        preds_geocell = []
        loss = 0 if self.training else None

        # Loop over every data sample
        for i, (emb, candidates, c_probs) in enumerate(
            zip(embedding, candidate_cells, candidate_probs)
        ):
            top_preds = []
            top_distances = []

            # Loop over every candidate cell
            for cell in candidates[: self.topk]:
                # Embedding distance
                cell_id = cell.item()
                # if cell_id in [121, 650, 1859]:  # TODO: fix
                #     cell_id = 1436

                cell_emb = self.protos[
                    cell_id
                ]  # TODO:cell_emb: Tensor of shape (num_protos, dim_embedding), or None. dim_embedding:emb of pictures in that geocell
                if cell_emb is None:
                    if self.verbose:
                        print(f"Proto dataset for geocell {cell_id} is None.")

                    top_distances.append(torch.tensor(-100000, device="cuda"))
                    top_preds.append([0.0, 0.0])
                    continue

                cell_emb = cell_emb["embedding"].to("cuda")
                logits = -self._euclidean_distance(cell_emb, emb)

                # Get top cluster and corresponding coordinates
                top_distances.append(torch.max(logits).item())
                pred_cluster_id = torch.argmax(logits, dim=-1)
                entry = self.protos[
                    cell_id
                ][
                    pred_cluster_id.item()
                ]  # TODO: dont think this will work. Need a way to get the data for the cluster. Change this to use self.proto_df. The entry is like a row in proto_df
                # entry: {"indices":list[int], "count":int, "centroid_lat":float, "centroid_lng":float}
                lng, lat = self._within_cluster_refinement(emb, entry)
                top_preds.append([lng, lat])

            # Temperature softmax over cluster candidates
            top_distances = torch.tensor(top_distances, device="cuda")
            probs = self._temperature_softmax(top_distances)

            # Multiply proto probabilities with geocell probabilities
            initial_guess = torch.argmax(c_probs[: self.topk]).item()
            final_probs = c_probs[: self.topk] * probs
            refined_guess = torch.argmax(final_probs).item()
            if refined_guess != initial_guess and self.verbose:
                print("\t\tRefinement changed geocell.")

            # Don't refine if refinement is more than max_refinement km
            refined_LLH = torch.tensor(top_preds[refined_guess], device="cuda")
            refined_LLH = refined_LLH.unsqueeze(0)
            initial_LLH = initial_preds[i].unsqueeze(0)
            distance = haversine(initial_LLH, refined_LLH)[0]
            if distance > self.max_refinement:
                final_probs = c_probs[: self.topk]
                if self.verbose:
                    print("\t\tCancelled refinement: distance too far.")

            final_pred_id = torch.argmax(final_probs).item()
            guess_index.append(final_pred_id)
            preds_LLH.append(top_preds[final_pred_id])
            preds_geocell.append(candidates[final_pred_id])

        # Look at percent of changed predictions
        guess_index = torch.tensor(guess_index, device="cuda")
        perc_changed = (guess_index != 0).sum() / guess_index.size(0)
        print(f"Changed geocell predictions of {perc_changed * 100:.1f} % of guesses.")

        preds_LLH = torch.tensor(preds_LLH, device="cuda")
        preds_geocell = torch.tensor(preds_geocell, device="cuda")
        return loss, preds_LLH, preds_geocell

    def _within_cluster_refinement(
        self, emb: Tensor, cluster: Dict[str, Tensor]
    ) -> Tuple[float, float]:
        """Refines the guess even further by picking the image in a cluster that matches the best.

        Args:
            emb (Tensor): embedding of query image.
            cluster (Dict[str, Tensor]): Huggingface dataset entry.

        Returns:
            Tuple[float, float]: (lng, lat)
        """
        if cluster["count"] == 0:
            return cluster["centroid_lng"].item(), cluster["centroid_lat"].item()

        points = self.dataset["train"][
            cluster["indices"]
        ]  # TODO: query the database for these indices directly
        embeddings = points["embedding"].to("cuda")  # or points["emb_tiny_vit"]
        # if embeddings.dim() == 3 and embeddings.size(1) == 4:
        if (
            embeddings.dim() == 3 and embeddings.size(1) == 4
        ):  # TODO:check this later with data
            embeddings = embeddings.mean(dim=1)

        distances = self._euclidean_distance(embeddings, emb)
        max_index = torch.argmax(distances).item()
        max_point = points["labels"][
            max_index
        ]  # TODO: get the lng and lat by the cluster directly?
        return max_point[0].item(), max_point[1].item()

    def load_prototypes(self) -> List[Dataset]:
        """Load prototypes used for matching.

        Returns:
            List[Dataset]: list of datasets for every geocell
        """
        print("Initializing ProtoRefiner. This might take a while ...")

        # Create progress bar
        progress_bar = tqdm(
            total=self.num_geocells,
            desc="Processing",
            position=0,
            leave=True,
            file=sys.stdout,
        )
        disable_progress_bar()  # dataset.map progress bar, not tqdm

        # Multi-processing for CPU-bound (not I/O bound) prototype generation
        with ProcessPoolExecutor(max_workers=None) as executor:
            future_to_index = {
                executor.submit(self._get_prototypes, i): i
                for i in range(self.num_geocells)
            }
            self.protos = [None] * self.num_geocells
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    self.protos[index] = future.result()
                except Exception as exc:
                    print(f"Task {index} generated an exception: {exc}")

                progress_bar.update(1)

        # Close progress bar after completion
        progress_bar.close()
        enable_progress_bar()

        print("Initialization of ProtoRefiner complete.")

    def _get_prototypes(self, cell: int) -> Dataset:
        """Gets embedding and geo-tensor prototypes for cell.

        Args:
            cell (int): geocell index

        Returns:
            Dataset: dataset including prototypes
        """
        try:
            cell_df = self.proto_manager.get_indices_for_cell(
                cell
            )  # Returns DataFrame with an 'indices' column (list[int]) per row

        # Some geocells might overlap with others, causing no data points to be in the cell
        except KeyError:
            return None

        # Normalize to DataFrame with an 'indices' column
        if isinstance(cell_df, list):
            # If a raw list of ints is returned, wrap into a single-row DataFrame
            cell_df = pd.DataFrame({"indices": [cell_df]})
        if type(cell_df) is pd.core.series.Series:
            cell_df = pd.DataFrame([cell_df])

        if len(cell_df["indices"]) == 0:
            return None

        data = Dataset.from_pandas(cell_df)
        # Data is a list of all the indices in that geocell
        data = data.map(self._compute_protos_for_cell)
        data.set_format("torch")
        return data  # This becomes a dataset with all the prototypes for that geocell. {"embedding": proto_emb}. proto_emb is a tensor of shape (dim_embedding,number of protos)

    def _cosine_similarity(self, matrix: Tensor, vector: Tensor) -> Tensor:
        """Computes the cosine similarity between all vectors in matrix and vector.

        Args:
            matrix (Tensor): matrix of shape (N, dim_vector)
            vector (Tensor): vector of shape (dim_vector)

        Returns:
            Tensor: cosine similarities
        """
        dot_product = torch.mm(matrix, vector.unsqueeze(1))
        matrix_norm = torch.norm(matrix, dim=1).unsqueeze(1)
        vector_norm = torch.norm(vector)
        cosine_similarities = dot_product / (matrix_norm * vector_norm)

        return cosine_similarities.flatten()

    def _euclidean_distance(self, matrix: Tensor, vector: Tensor) -> Tensor:
        """Computes the euclidian distance between all vectors in matrix and vector.

        Args:
            matrix (Tensor): matrix of shape (N, dim_vector)
            vector (Tensor): vector of shape (dim_vector)

        Returns:
            Tensor: euclidean distances
        """
        v = vector.unsqueeze(0)
        distances = torch.cdist(matrix, v)
        return distances.flatten()

    def _temperature_softmax(self, input: Tensor) -> Tensor:
        """Performs softmax with learnable temperature.

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        ex = torch.exp(input / self.temperature)
        sum = torch.sum(ex, axis=0)
        return ex / sum

    def _compute_protos_for_cell(self, indices: List[int]) -> Dict:
        """Computes embedding and geo prototypes.

        Args:
            indices (List): data sample
            geo (bool, optional): whether geo tensor should be included.
                Defaults to False.

        Returns:
            Dict: modified data sample
        """

        emb = self.embedder.generate_embeddings(indices)
        return {"clip": emb["clip"], "tiny_vit": emb["tinyvit"]}


class Embeddings:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        df = load_sqlite_panorama_dataset(None)
        self.data = LocalGeoMapDataset(df)

        print("Loading embedder models...")
        self.model_clip = CLIPEmbedding(
            model_name="", device=self.device, load_checkpoint=False, panorama=False
        )
        self.model_tiny = TinyViTEmbedding(
            model_name="tiny_vit_21m_512.dist_in22k_ft_in1k",
            device=self.device,
            load_checkpoint=False,
            panorama=False,
        )
        # Robustly load lat/lon mapping; tolerate whitespace and skip malformed lines
        try:
            df_latlon = pd.read_csv(
                "data/out/sv_points_all_latlong.txt",
                header=None,
                names=["lat", "lon"],
                usecols=[0, 1],
                sep=",",
                on_bad_lines="skip",
                engine="python",
                skip_blank_lines=True,
                skipinitialspace=True,  # handles "lat, lon" with a space after comma
            )
            # Coerce to float and drop rows with NaNs
            df_latlon = df_latlon.apply(pd.to_numeric, errors="coerce").dropna()
            self.lat_lon_by_index = df_latlon.to_numpy(copy=False)
        except Exception:
            # Fallback to numpy with tolerant parsing
            self.lat_lon_by_index = np.genfromtxt(
                "data/out/sv_points_all_latlong.txt",
                delimiter=",",
                usecols=(0, 1),
                dtype=float,
                filling_values=np.nan,
                autostrip=True,
            )
        print("Embedder models loaded.")

    def generate_embeddings(self, indices: List[int]):
        clip_emb_list: List[torch.Tensor] = []
        tiny_emb_list: List[torch.Tensor] = []

        with torch.no_grad():
            for idx in indices:
                # Guard against out-of-range and malformed rows
                if idx < 0 or idx >= len(self.lat_lon_by_index):
                    continue
                lat, lon = self.lat_lon_by_index[idx]
                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue
                d = {"lat": float(lat), "lon": float(lon)}
                pictures = self.data.get_tensor_of_panorama_images_from_point(d)

                clip = self.model_clip._get_embedding(pictures)
                tiny = self.model_tiny._get_embedding(pictures)

                # If model returns (V, D), average across views to get (D,)
                if clip.dim() == 2:
                    clip = clip.mean(dim=0)
                if tiny.dim() == 2:
                    tiny = tiny.mean(dim=0)

                # If model returns (D,), keep as is. If returns (V, D, ...), collapse to (D,)
                if clip.dim() > 2:
                    clip = clip.flatten(start_dim=0).mean(dim=0)
                if tiny.dim() > 2:
                    tiny = tiny.flatten(start_dim=0).mean(dim=0)

                clip_emb_list.append(clip)
                tiny_emb_list.append(tiny)

        # Stack into (N, D). This is compatible with:
        # embeddings = entries["embedding"]
        # if embeddings.dim() == 3: embeddings = embeddings.mean(dim=1)
        # proto_emb = embeddings.mean(dim=0)
        clip_tensor = (
            torch.stack(clip_emb_list, dim=0)
            if clip_emb_list
            else torch.empty(0, 0, device=self.device)
        )
        tiny_tensor = (
            torch.stack(tiny_emb_list, dim=0)
            if tiny_emb_list
            else torch.empty(0, 0, device=self.device)
        )

        if clip_tensor.dim() == 3:
            clip_tensor = clip_tensor.mean(dim=1)
        clip_tensor = clip_tensor.mean(dim=0)

        if tiny_tensor.dim() == 3:
            tiny_tensor = tiny_tensor.mean(dim=1)
        tiny_tensor = tiny_tensor.mean(dim=0)
        return {"tinyvit": tiny_tensor, "clip": clip_tensor}


if __name__ == "__main__":
    print("Initializing ProtoRefiner for generating prototypes...")
    proto = ProtoRefiner()
    print("Saving prototypes to disk")
    print("Exiting...")
