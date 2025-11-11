# adapters/tinyvit_adapter.py
from types import SimpleNamespace
from typing import Optional
import torch
import torch.nn as nn
import timm

try:
    # Optional: for a ready-to-use preprocessing pipeline
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform
except Exception:
    resolve_model_data_config = None
    create_transform = None


class TinyViTAdapter(nn.Module):
    """
    Wrap a timm TinyViT so it:
      - exposes .config with _name_or_path and hidden size info
      - returns an object with BOTH .pooler_output and .last_hidden_state
      - exposes .vision_model.encoder.layers (list of stage modules) for freezing
      - (optional) provides a build_transform() helper for preprocessing

    Works with SuperGuessr without any code changes.
    """

    def __init__(
        self,
        model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
        pretrained: bool = True,
        global_pool: str = "avg",
        features_only: bool = False,  # keep default False for pooled embeddings
    ):
        super().__init__()

        if features_only:
            # Not needed for your current SuperGuessr flow, but supported
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, features_only=True
            )
            # num_features of features_only models is list-like; use last stage dim
            hidden = getattr(self.backbone, "feature_info", None)
            if hidden is not None:
                hidden = hidden[-1]["num_chs"]
        else:
            # num_classes=0 + global_pool => (B, C) pooled embeddings
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool=global_pool,
            )
            hidden = getattr(self.backbone, "num_features", None) or getattr(
                self.backbone, "num_classes", None
            )

        # --- config shim ---
        # Keep both attributes so your _set_hidden_size() works either way:
        # - If it checks .config.hidden_size -> it will pick "transformer" path.
        # - If it falls back to .config.hidden_sizes[-1] -> "convnext" path.
        self.config = SimpleNamespace(
            hidden_size=hidden,  # enables "transformer" path if used
            hidden_sizes=[hidden],  # enables "convnext" path if used
            _name_or_path=model_name,
        )

        # --- encoder.layers shim for freezing ---
        # timm TinyViT has a .stages (Sequential). Expose as a simple list.
        stages = getattr(self.backbone, "stages", None)
        if stages is None:
            # Defensive: some timm variants may differ; fall back to children()
            stages = [m for m in self.backbone.children()]
        encoder = SimpleNamespace(layers=list(stages))
        self.vision_model = SimpleNamespace(encoder=encoder)

        # Track a fully-frozen flag for correct BN/LN behavior in .train()
        self._fully_frozen = False

    # -------- convenience methods (optional) --------
    def build_transform(self):
        """Return a torchvision transform matching this backbone (if timm data utils are available)."""
        if resolve_model_data_config is None or create_transform is None:
            raise RuntimeError(
                "timm data transforms not available in this environment."
            )
        cfg = resolve_model_data_config(self.backbone)
        return create_transform(**cfg)

    def freeze_all(self, eval_mode: bool = True):
        """Freeze all TinyViT params (useful when you already fine-tuned it)."""
        for p in self.parameters():
            p.requires_grad = False
        self._fully_frozen = True
        if eval_mode:
            super().train(False)  # set module to eval so BN/LN stats don't update
        return self

    def unfreeze_all(self):
        """Unfreeze all TinyViT params."""
        for p in self.parameters():
            p.requires_grad = True
        self._fully_frozen = False
        return self

    def freeze_all_but_last_stage(self):
        layers = list(self.vision_model.encoder.layers)  # stages list
        for m in layers[:-1]:
            for p in m.parameters():
                p.requires_grad = False
        return self

    def train(self, mode: bool = True):
        """
        Respect full-freeze mode: if fully frozen, keep the module in eval even if
        outer code sets .train(True). This prevents BN running-stats updates.
        """
        if self._fully_frozen:
            return super().train(False)
        return super().train(mode)

    # -------- forward --------
    def forward(
        self, pixel_values: torch.Tensor = None, x: Optional[torch.Tensor] = None
    ):
        """
        SuperGuessr calls base_model(pixel_values=...).
        We return an object with:
          - pooler_output: (B, C)
          - last_hidden_state: (B, 1, C)  # so your "transformer" path mean-pool works
        """
        if pixel_values is None and x is not None:
            pixel_values = x  # allow .forward(x=...) too

        out = self.backbone(pixel_values)

        # If features_only=True, timm returns a list of feature maps.
        # Aggregate to a single vector (global avg) so downstream stays identical.
        if isinstance(out, (list, tuple)):
            # Global average pool the last feature map
            last = out[-1]
            # last: (B, C, H, W) -> (B, C)
            out = last.mean(dim=(-2, -1))

        return SimpleNamespace(
            pooler_output=out,  # (B, C)
            last_hidden_state=out.unsqueeze(
                1
            ),  # (B, 1, C) to satisfy transformer-style path
        )
