import torch
from torch import Tensor
from typing import Dict, Callable
from PIL import Image
import timm


class TinyViTEmbedding(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
        device: str = "cuda",
        load_checkpoint: bool = False,
        panorama: bool = False,
    ):
        """TinyViT embedding model (not trainable)

        Args:
            model_name (str): TinyViT model version. Defaults to 'tiny_vit_21m_512.dist_in22k_ft_in1k'.
            device (str, optional): where to run the model. Defaults to 'cuda'.
            load_checkpoint (bool, optional): whether to load checkpoint from file.
                Defaults to False.
            panorama (bool): if four images should be embedded.
                Defaults to False.
        """
        super().__init__()
        self.device = device
        self.panorama = panorama
        self.model_name = model_name

        # Create TinyViT model
        self.tinyvit_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Return embeddings instead of class predictions
        )

        # Load checkpoint if required
        if load_checkpoint:
            state_dict = torch.load(model_name, map_location=torch.device("cuda"))
            self.tinyvit_model.load_state_dict(state_dict)
            print("Loaded embedder from checkpoint:", model_name)

        # Move model to device
        if isinstance(device, str):
            self.tinyvit_model = self.tinyvit_model.to(self.device)
        else:
            self.tinyvit_model = self.tinyvit_model.cuda(self.device)

        # Get model-specific transforms
        data_config = timm.data.resolve_model_data_config(self.tinyvit_model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        self.eval()

    def _get_embedding(self, image: Image) -> Tensor:
        """Computes embedding for a single image.

        Args:
            image (Image): PIL image or tensor

        Returns:
            Tensor: embedding
        """
        with torch.no_grad():
            if not isinstance(image, Tensor):
                # Image is PIL, apply transforms
                pixel_values = self.transforms(image).unsqueeze(0)
            else:
                # Already a tensor
                pixel_values = image

            # Move to device
            if isinstance(self.device, str):
                pixel_values = pixel_values.to(self.device)
            else:
                pixel_values = pixel_values.cuda(self.device)

            # Get embedding
            embedding = self.tinyvit_model(pixel_values)
            return embedding

    def _pre_embed_hook(self) -> Callable:
        """Hook to store forward activations of a specific layer.

        Returns:
            Callable: The hook to be registered on a module's forward function.
        """

        def hook(model, input, output):
            self.pre_embed_outputs = output[0]

        return hook

    def forward(self, image: Tensor | Dict, **kwargs) -> torch.Tensor:
        """Computes forward pass to generate embeddings.

        Args:
            image (Tensor | Dict): Single image or first image in panorama
            **kwargs: Additional images for panorama (image_2, image_3, image_4)

        Returns:
            Tensor: Embedding tensor. Shape (batch, embed_dim) for single image,
                   (batch, 4, embed_dim) for panorama.
        """
        if isinstance(image, Tensor):
            return self._get_embedding(image)

        if "image_2" not in kwargs:
            return self._get_embedding(image)

        # Panorama mode: process 4 images
        embeddings = []

        # First image from positional parameter
        embedding = self._get_embedding(image)
        embeddings.append(embedding)

        # Remaining images from kwargs
        for col in ["image_2", "image_3", "image_4"]:
            embedding = self._get_embedding(kwargs[col])
            embeddings.append(embedding)

        return torch.stack(embeddings, dim=1)
