import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import os
from models.layers.positional_encoder import PositionalEncoder
from models.utils import ModelOutput, haversine_matrix, smooth_labels
from config import CLIP_PRETRAINED_HEAD, CLIP_EMBED_DIM
from data.geocells.geocell_manager import GeocellManager
import pandas as pd


# Constants
NUM_ATTENTION_HEADS = 16

GEOGUESSR_HEADING_SINGLE = [0.0, 1.0]
GEOGUESSR_HEADING_MULTI = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]


class SuperGuessr(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        panorama: bool = False,
        hierarchical: bool = False,
        should_smooth_labels: bool = False,
        serving: bool = False,
        freeze_base: bool = False,
        num_candidates: int = 5,
        embed_dim: int = CLIP_EMBED_DIM,
        **kwargs,
    ):
        """Initializes a location prediction model classification model.

        Args:
            base_model (nn.Module): vision encoder model on top of which the location
                predictor is built. If None, assumes that model is run directly on embeddings.
            panorama (bool, optional): whether four images are passed in as a panorama.
                Defaults to False.
            hierarchical (bool, optional): whether to use a hierarchical model to combine embeddings.
                Defaults to False.
            should_smooth_labels (bool, optional): If labels should be smoothed. Label smoothing
                works only in classification mode and penalizes guesses based on the actual distance
                of the geocell to the correct location instead of penalizing equally across all
                incorrect cells. Label smoothing also makes the prediction task easier.
                Defaults to False.
            serving (bool, optional): Whether model is instantiated for serving purposes only. If set to
                True, outputs solely lng/lat predictions in eval mode.
            freeze_base (bool, optional): If the weights of the base model should be frozen.
                Defaults to False.
            num_candidates (int, optional): Number of geocell candidates to
                produce for refinement. Defaults to 5.
            embed_dim (int, optional): Embedding dimension if base model is None. Defaults to 1024.
        """
        super(SuperGuessr, self).__init__()

        # List kwargs
        if len(kwargs) > 0:
            print(f"Not using keyword arguments: {list(kwargs.keys())}")

        # Save variables
        self.base_model = base_model
        self.panorama = panorama
        self.hidden_size = embed_dim
        self.serving = serving
        self.should_smooth_labels = should_smooth_labels
        self.freeze_base = freeze_base
        self.hierarchical = hierarchical
        self.num_candidates = num_candidates

        # Setup
        self._set_hidden_size()
        geocell_dir = (
            "data/geocells/finished_geocells"  # must be a DIRECTORY containing *.pickle
        )
        self._geocell_mgr = GeocellManager(geocell_dir)
        # Prefer proto_df.csv ordering via geocell_index if available
        centroids = _build_centroids_from_proto_df("data/geocells/proto_df.csv")
        if centroids is None:
            centroids = _build_centroids_from_manager(self._geocell_mgr)

        # (num_cells, 2) in (lng, lat)
        self.geocell_centroid_coords = nn.Parameter(centroids, requires_grad=False)
        self.num_cells = centroids.size(0)

        # Input dimension for cell layer
        self.input_dim = self.hidden_size

        # Self-attention layer
        if self.hierarchical:
            print("Number of attention heads:", NUM_ATTENTION_HEADS)
            self.heading_pad = 0
            self.pos_encoder = PositionalEncoder(self.input_dim + self.heading_pad)
            self.self_attn = nn.MultiheadAttention(
                self.input_dim + self.heading_pad,
                NUM_ATTENTION_HEADS,
                dropout=0.1,
                batch_first=True,
            )
            self.relu = nn.ReLU()

        # Cell layer
        self.cell_layer = nn.Linear(self.input_dim, self.num_cells)
        self.softmax = nn.Softmax(dim=-1)

        # Freeze / load parameters
        self._freeze_params()

        # Loss
        self.loss_fnc = nn.CrossEntropyLoss()
        print(
            f"Initialized SuperGuessr classification model with {self.num_cells} geocells."
        )

    def _set_hidden_size(self):
        """
        Determines the hidden size of the model
        """
        if self.base_model is not None:
            try:
                self.hidden_size = self.base_model.config.hidden_size
                self.mode = "transformer"

            except AttributeError:
                self.hidden_size = self.base_model.config.hidden_sizes[-1]
                self.mode = "convnext"

    def _freeze_params(self):
        """Freezes model parameters depending on mode"""
        if self.base_model is not None:
            if self.freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False

            # Load parameters and freeze relevant parameters
            elif (
                "clip-vit" in self.base_model.config._name_or_path and not self.serving
            ):
                head = CLIP_PRETRAINED_HEAD
                if os.path.exists(head):
                    self.load_state(head)
                    print(f"Initialized model parameters from model: {head}")
                    for param in self.base_model.vision_model.encoder.layers[
                        :-1
                    ].parameters():
                        param.requires_grad = False
                else:
                    print(
                        f"Warning: pretrained head not found at '{head}'. "
                        "Proceeding without loading and without freezing base layers."
                    )

            elif "tiny" in self.base_model.config._name_or_path and not self.serving:
                self.base_model.freeze_all_but_last_stage()

    def load_state(self, path: str):
        """Loads weights from path and applies them to the model.

        Args:
            path (str): path to model weights
        """
        own_state = self.state_dict()
        # Always load on CPU; let the caller/accelerate move to device
        state_dict = torch.load(path, map_location="cpu")
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"Parameter {name} not in model's state.")
                continue

            if isinstance(param, Parameter):
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception as e:
                print(f"Failed loading parameter {name}: {e}")

    def _assert_requirements(
        self,
        pixel_values: Tensor = None,
        embedding: Tensor = None,
    ):
        """Checks assertions related to input.

        Args:
            pixel_values (Tensor, optional): preprocessed images pixel values.
            embedding (Tensor, optional): image embeddings if no pass through
                a base model is performed.
        """

        if self.base_model is not None:
            assert pixel_values is not None, (
                'Parameter "pixel_values" must be supplied if model has a base model.'
            )
        else:
            assert embedding is not None, (
                'Parameter "embedding" must be supplied if model '
                "does not have a base model."
            )

    def _to_one_hot(self, tensor: Tensor) -> Tensor:
        """Convert a scalar tensor to a one-hot encoded tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            Tensor: The one-hot encoded tensor.
        """
        if tensor.dim() == 0:
            one_hot = torch.zeros(self.num_cells, device=tensor.device)
            one_hot[tensor.item()] = 1
            return one_hot
        else:
            return tensor

    def forward(
        self,
        pixel_values: Tensor = None,
        embedding: Tensor = None,
        labels: Tensor = None,
        labels_clf: Tensor = None,
        index: Tensor = None,
    ) -> ModelOutput:
        """Computes forward pass through network.

        Args:
            pixel_values (Tensor, optional): preprocessed images pixel values.
            embedding (Tensor, optional): image embeddings if no pass through
                a base model is performed.
            labels (Tensor, optional): coordinates or classification labels.
            labels_clf (Tensor, optional): index of ground truth geocell.

        Returns:
            ModelOutput: named tuple of model outputs. If serving, will be tuple.

        Note:
            If a base model was specified, pixel_values need to be supplied.
            Otherwise, if the model is working directly on embeddings,
            embedding must not be None.
        """
        self._assert_requirements(pixel_values, embedding)

        # If panorama, (N, 4, C, H, W) -> (N * 4, C, H, W)
        if self.panorama and pixel_values is not None:
            # Expect pixel_values as (B, 4, C, H, W)
            assert pixel_values.dim() == 5, "panorama=True expects (B, 4, C, H, W)"
            n, v, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(n * v, c, h, w)

        # Feed through base model
        if self.base_model is not None and pixel_values is not None:
            if pixel_values.dim() > 4:
                pixel_values = pixel_values.squeeze(1)

            outs = self.base_model(pixel_values=pixel_values)

            # Accept HF-style outputs OR a raw tensor from timm
            if hasattr(outs, "last_hidden_state") and self.mode == "transformer":
                # (B, T, C) -> mean over tokens
                embedding = outs.last_hidden_state.mean(dim=1)
            elif hasattr(outs, "pooler_output"):
                embedding = outs.pooler_output  # (B, C)
            else:
                # timm(num_classes=0) returns a (B, C) tensor
                embedding = outs

            # (N * 4, C) -> (N, 4, C) for panoramas
            if self.panorama:
                embedding = embedding.view(n, v, -1)

        layer_input = embedding

        # Handle four image input
        if self.panorama:
            if self.hierarchical:
                layer_input = self.pos_encoder(layer_input)
                output = self.self_attn(
                    layer_input, layer_input, layer_input, need_weights=False
                )[0]
                output = output[:, 0]
            else:
                output = layer_input.mean(dim=1)  # (N, 4, C) -> (N, C)

        # Single image
        else:
            output = embedding  # (N, C)

        # Linear layer
        logits = self.cell_layer(output)
        geocell_probs = self.softmax(logits)

        # Compute coordinate prediction
        geocell_preds = torch.argmax(geocell_probs, dim=-1)
        pred_centroid_coordinate = torch.index_select(
            self.geocell_centroid_coords.data, 0, geocell_preds
        )
        # label_probs = self._to_one_hot(labels_clf)  # labels_clf if normal

        # Get top 'num_candidates' geocell candidates
        geocell_topk = torch.topk(geocell_probs, self.num_candidates, dim=-1)

        # Serving (inference)
        if not self.training and self.serving:
            return pred_centroid_coordinate, geocell_topk, embedding

        # Soft labels based on distance
        if getattr(self, "should_smooth_labels", False) and labels is not None:
            # distances: (B, num_cells)
            distances = haversine_matrix(labels, self.geocell_centroid_coords.data.t())
            # unnormalized similarities -> normalize to probability distribution
            soft_targets = smooth_labels(distances)
            soft_targets = soft_targets / soft_targets.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            # soft cross-entropy with log-softmax
            log_probs = F.log_softmax(logits, dim=-1)
            loss_clf = -(soft_targets * log_probs).sum(dim=-1).mean()
        else:
            # standard hard-label CE expects class indices
            loss_clf = self.loss_fnc(logits, labels_clf)
        loss = loss_clf

        # Results
        output = ModelOutput(
            loss,
            loss_clf,
            pred_centroid_coordinate,
            geocell_preds,
            geocell_topk,
            embedding,
        )
        return output

    def __str__(self):
        rep = "SuperGuessr(\n"
        rep += f"\tbase_model\t= {self.base_model is not None}\n"
        rep += f"\tpanorama\t= {self.panorama}\n"
        rep += f"\thierarchical\t= {self.hierarchical}\n"
        rep += f"\tembedding_size\t= {self.hidden_size}\n"
        rep += f"\tinput_dim\t= {self.input_dim}\n"
        rep += f"\tnum_geocells\t= {self.num_cells}\n"
        rep += f"\tlabel_smoothing\t= {self.should_smooth_labels}\n"
        rep += f"\tfreeze_base\t= {self.freeze_base}\n"
        rep += f"\tserving\t\t= {self.serving}\n"
        rep += ")"
        return rep


def _centroid_from_points(cell):
    # crude mean; replace with your preferred centroid logic if you have one
    lngs = [p["longitude"] for p in cell.points]
    lats = [p["latitude"] for p in cell.points]
    if len(lngs) == 0:
        return (0.0, 0.0)
    return (sum(lngs) / len(lngs), sum(lats) / len(lats))


def _build_centroids_from_manager(mgr: GeocellManager):
    """
    Returns:
      centroids: (num_cells, 2) float32 tensor in (lng, lat)
    """
    rows = []
    for country in mgr.geocells:
        for adm1 in mgr.geocells[country]:
            for cell in mgr.geocells[country][adm1]:
                # prefer centroid attribute if present
                cen = getattr(cell, "centroid", None)
                if cen is None:
                    lng, lat = _centroid_from_points(cell)
                else:
                    # handle dict or tuple; ensure (lng, lat) order
                    if isinstance(cen, dict):
                        lng, lat = cen["longitude"], cen["latitude"]
                    else:
                        lng, lat = cen[0], cen[1]
                rows.append(
                    (str(country), str(adm1), cell.id, float(lng), float(lat), cell)
                )

    # deterministic ordering
    rows.sort(key=lambda r: (r[0], r[1], str(r[2])))

    centroids = []
    for idx, (country, adm1, geocell_id, lng, lat, _cell) in enumerate(rows):
        centroids.append([lng, lat])  # (lng, lat)

    centroids = torch.tensor(centroids, dtype=torch.float32)
    return centroids


def _build_centroids_from_proto_df(csv_path: str):
    """
    Build centroids ordered by geocell_index from proto_df.csv.
    Returns:
      centroids: (num_cells, 2) float32 tensor in (lng, lat) or None if csv not found.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Support legacy column name fallback
    idx_col = (
        "geocell_index"
        if "geocell_index" in df.columns
        else ("geocell_id" if "geocell_id" in df.columns else None)
    )
    if idx_col is None:
        return None

    # Deduplicate to one row per geocell_index (there may be multiple rows per clusters)
    df = df.sort_values(by=[idx_col])
    dedup = df.drop_duplicates(subset=[idx_col], keep="first")

    # Ensure required centroid columns exist
    if not {"centroid_lng", "centroid_lat"}.issubset(dedup.columns):
        return None

    centroids = torch.tensor(
        dedup[["centroid_lng", "centroid_lat"]].values, dtype=torch.float32
    )
    return centroids
