import torch
import pandas as pd
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from preprocessing import haversine_matrix, smooth_labels
from models.layers import PositionalEncoder
from models.utils import ModelOutput
from config import CLIP_PRETRAINED_HEAD, CLIP_EMBED_DIM, GEOCELL_PATH


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
        geocell_path = GEOCELL_PATH
        self.lla_geocells = self.load_geocells(geocell_path)
        self.num_cells = self.lla_geocells.size(0)

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
                self.load_state(head)
                print(f"Initialized model parameters from model: {head}")
                for param in self.base_model.vision_model.encoder.layers[
                    :-1
                ].parameters():
                    param.requires_grad = False

            elif (
                "tiny-vit" in self.base_model.config._name_or_path and not self.serving
            ):
                head = CLIP_PRETRAINED_HEAD  # TODO
                self.load_state(head)
                print(f"Initialized model parameters from model: {head}")
                for param in self.base_model.vision_model.encoder.layers[
                    :-1
                ].parameters():
                    param.requires_grad = False

    def load_geocells(self, path: str) -> Tensor:
        """Loads geocell centroids and converts them to ECEF format

        Args:
            path (str, optional): path to geocells. Defaults to GEOCELL_PATH.

        Returns:
            Tensor: ECEF geocell centroids
        """
        geo_df = pd.read_csv(path)
        lla_coords = torch.tensor(geo_df[["lng", "lat"]].values)
        lla_geocells = nn.parameter.Parameter(data=lla_coords, requires_grad=False)
        return lla_geocells

    def _move_to_cuda(
        self,
        pixel_values: Tensor = None,
        embedding: Tensor = None,
        labels: Tensor = None,
        labels_clf: Tensor = None,
    ):
        """Moves supplied tensors to device.

        Args:
            pixel_values (Tensor, optional): preprocessed images pixel values.
            embedding (Tensor, optional): image embeddings if no pass through
                a base model is performed.
            labels (Tensor, optional): coordinates or classification labels.
            labels_clf (Tensor, optional): index of ground truth geocell.
        """
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        if not self.training and device == "cuda":
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            if embedding is not None:
                embedding = embedding.to(device)

            if labels is not None:
                labels = labels.to(device)

            if labels_clf is not None:
                labels_clf = labels_clf.to(device)

        return (
            pixel_values,
            embedding,
            labels,
            labels_clf,
        )

    def load_state(self, path: str):
        """Loads weights from path and applies them to the model.

        Args:
            path (str): path to model weights
        """
        own_state = self.state_dict()
        state_dict = torch.load(path, map_location=torch.device("cuda"))
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"Parameter {name} not in model's state.")
                continue

            if isinstance(param, Parameter):
                param = param.data

            own_state[name].copy_(param)

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
                'Parameter "pixel_values" must be supplied if model \
                                              has a base model.'
            )
        else:
            assert embedding is not None, (
                'Parameter "embedding" must be supplied if model \
                                           does not have a base model.'
            )

    def _to_one_hot(self, tensor: Tensor) -> Tensor:
        """Convert a scalar tensor to a one-hot encoded tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            num_classes (int): The number of classes for one-hot encoding.

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

        # Device
        (
            pixel_values,
            embedding,
            labels,
            labels_clf,
        ) = self._move_to_cuda(
            pixel_values,
            embedding,
            labels,
            labels_clf,
        )

        # If panorama, (N, 4 * pixels) -> (N * 4, pixels)
        if self.panorama and pixel_values is not None:
            num_samples = pixel_values.size(0)
            pixel_values = pixel_values.reshape((num_samples * 4, 3, 336, 336))

        # Feed through base model
        if self.base_model is not None and pixel_values is not None:
            if pixel_values.dim() > 4:
                pixel_values = pixel_values.squeeze(1)

            embedding = self.base_model(pixel_values=pixel_values)
            if self.mode == "transformer":
                embedding = embedding.last_hidden_state
                embedding = torch.mean(embedding, dim=1)

            else:
                embedding = embedding.pooler_output

            # (N * 4, pixels) -> (N, 4, pixels)
            if self.panorama:
                embedding = embedding.reshape((num_samples, 4, -1))

        layer_input = embedding

        # Handle four image input
        if self.panorama:
            # Hierarchical architecture
            if self.hierarchical:
                # Positional encoding
                layer_input = self.pos_encoder(layer_input)

                # Multi-head self attention
                output = self.self_attn(
                    layer_input, layer_input, layer_input, need_weights=False
                )[0]

                # Pool (CLS) and remove zero concats
                output = output[:, 0]

            # Average embeddings
            else:
                output = layer_input.mean(dim=1)

        # Single Image
        elif layer_input.size(1) == 4:
            output = embedding[:, 0]

        else:
            output = embedding

        # Linear layer
        logits = self.cell_layer(output)
        geocell_probs = self.softmax(logits)

        # Compute coordinate prediction
        geocell_preds = torch.argmax(geocell_probs, dim=-1)
        pred_LLH = torch.index_select(self.lla_geocells.data, 0, geocell_preds)
        label_probs = self._to_one_hot(labels_clf)  # labels_clf if normal

        # Get top 'num_candidates' geocell candidates
        geocell_topk = torch.topk(geocell_probs, self.num_candidates, dim=-1)

        # Serving
        if not self.training and self.serving:
            return pred_LLH, geocell_topk, embedding

        # Soft labels based on distance
        if self.should_smooth_labels:
            distances = haversine_matrix(labels, self.lla_geocells.data.t())
            label_probs = smooth_labels(distances)

        # Loss
        loss_clf = self.loss_fnc(logits, label_probs)
        loss = loss_clf

        # Results
        output = ModelOutput(
            loss,
            loss_clf,
            pred_LLH,
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
