"""Functions and classes for loading and handling models conveniently."""
import contextlib
from einops import repeat
from typing import Union, Dict, Optional

import torch
from torch import Tensor

from cheff.ldm.models.autoencoder import AutoencoderKL
from cheff.ldm.models.diffusion.ddpm import LatentDiffusion

from cheff.ldm.models.diffusion.ddpm_tab import LatentDiffusion as LatentDiffusion_tab
from cheff.ldm.models.diffusion.ddim import DDIMSampler


class CheffAEModel:
    def __init__(self, model_path: str, device: Union[str, int, torch.device] = "cuda") -> None:
        self.device = device

        with contextlib.redirect_stdout(None):
            self.model = AutoencoderKL(
                embed_dim=3,
                ckpt_path=model_path,
                ddconfig={
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                lossconfig={"target": "torch.nn.Identity"},
            )
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return self.model.encode(x).mode()

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class CheffLDM:
    def __init__(self, model_path: str, ae_path: Optional[str] = None, device: Union[str, int, torch.device] = "cuda") -> None:
        self.device = device
        with contextlib.redirect_stdout(None):
            self.model = self._init_checkpoint(model_path, ae_path)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.out_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

    @torch.no_grad()
    def sample(
        self, batch_size: int = 1, sampling_steps: int = 100, eta: float = 1.0, decode: bool = True, *args, **kwargs
    ) -> Tensor:
        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(sampling_steps, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False)

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion:
        config_dict = self._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        state_dict = torch.load(model_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
        assert len(missing) == 0 and len(unexpected) == 0
        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "image_size": 64,
            "channels": 3,
            "monitor": "val/loss_simple_ema",
            "unet_config": CheffLDM._get_unet_config_dict(),
            "first_stage_config": CheffLDM._get_first_stage_config_dict(ae_path),
            "cond_stage_config": "__is_unconditional__",
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 3, 4],
                "num_head_channels": 32,
            },
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "target": "cheff.ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 3,
                "ckpt_path": ae_path,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        }


class CheffLDMT2I(CheffLDM):
    @torch.no_grad()
    def sample(
        self,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        *args,
        **kwargs,
    ) -> Tensor:
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps, conditioning=conditioning, batch_size=1, shape=self.sample_shape, eta=eta, verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "cond_stage_key": "caption",
            "image_size": 64,
            "channels": 3,
            "cond_stage_trainable": True,
            "conditioning_key": "crossattn",
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "unet_config": CheffLDMT2I._get_unet_config_dict(),
            "first_stage_config": CheffLDMT2I._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMT2I._get_cond_config_dict(),
        }

    @staticmethod
    def _get_cond_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.encoders.modules.BERTEmbedder",
            "params": {
                "n_embed": 1280,
                "n_layer": 32,
            },
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_heads": 8,
                "use_spatial_transformer": True,
                "transformer_depth": 1,
                "context_dim": 1280,
                "use_checkpoint": True,
                "legacy": False,
            },
        }


class CheffLDMTab2IAdaptor:
    def __init__(
        self,
        model_path: str,
        ae_path: Optional[str] = None,
        device: Union[str, int, torch.device] = "cuda",
        num_layer=12,
        max_event_len=2048,
        n_embed=256,
        context_dim=1280,
        condition_feat_dim=1024,
        condition_type=None,
        conditioning_key=None,
        tab_encoder_architecture="TransformertabflatWrapper",
    ):
        self.device = device
        self.num_layer = num_layer
        self.max_event_len = max_event_len
        self.n_embed = n_embed
        self.context_dim = context_dim
        self.condition_feat_dim = condition_feat_dim
        self.condition_type = condition_type
        self.conditioning_key = conditioning_key
        self.tab_encoder_architecture = tab_encoder_architecture
        assert self.tab_encoder_architecture in ["TabAdaptor", "ClipAdaptor", "ClipTabAdaptor", "ImgAdaptor", "VAEAdaptor", "MultiModalTransformerAdaptor", "ClipImgAdaptor"]
        # TabAdaptor: CLIP tab + VAE / VAE: VAE only /  ImgAdaptor: CLIP image + VAE / MultiModal: CLIP tab + img + VAE
        # ClipTabAdaptor: CLIP tab only / ClipAdaptor: clip tab + img /  ImgAdaptor: CLIP image + VAE
        
        with contextlib.redirect_stdout(None):
            self.model = self._init_checkpoint(model_path, ae_path)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.out_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

    @torch.no_grad()
    def sample(
        self,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: dict = None,
        batch_size: int = 1,
        *args,
        **kwargs,
    ) -> Tensor:
        # conditioning = self.model.get_learned_conditioning(conditioning)
        if isinstance(conditioning, dict):
            conditioning["c_crossattn"] = {k: v.to(self.device) for k, v in conditioning["c_crossattn"].items()}
            conditioning["c_crossattn"] = [self.model.get_learned_conditioning(conditioning["c_crossattn"])]
            if "c_concat" in conditioning:
                conditioning["c_concat"] = [self.model.encode_first_stage(conditioning["c_concat"].to(self.device)).mode().detach()]
        else:
            conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        x_T = None

        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=batch_size,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            x_T=x_T,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion_tab:
        config_dict = self._get_config_dict(self, ae_path)
        model = LatentDiffusion_tab(**config_dict)
        print(f"Loading checkpoint from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)
        return model

    @staticmethod
    def _get_config_dict(self, ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "",
            "cond_stage_key": self.condition_type,
            "image_size": 64,
            "channels": 3,
            "cond_stage_trainable": True,
            "conditioning_key": self.conditioning_key,
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "unet_config": CheffLDMTab2IAdaptor._get_unet_config_dict(self),
            "first_stage_config": CheffLDMTab2IAdaptor._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMTab2IAdaptor._get_cond_config_dict(self, ae_path),
        }

    @staticmethod
    def _get_cond_config_dict(self, ae_path) -> Dict:
        return {
            "target": f"cheff.ldm.modules.encoders.modules.{self.tab_encoder_architecture}",
            "params": {
                "autoencoder_config": CheffLDMTab2IAdaptor._get_first_stage_config_dict(ae_path)["params"],
                "clip_enc_checkpoint": None,
                "context_dim": self.context_dim,
                "max_event_len": self.max_event_len,
                "condition_feat_dim": self.condition_feat_dim,
                "clip_visual_enc_config": {
                    "input_resolution": 256,
                    "layers": 12,
                    "width": 768,
                    "patch_size": 32,
                    "heads": 12
                },
            },
        }

    @staticmethod
    def _get_unet_config_dict(self) -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3 if self.conditioning_key == "crossattn" else 6,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_heads": 8,
                "use_spatial_transformer": True,
                "transformer_depth": 1,
                "context_dim": self.context_dim,
                "use_checkpoint": True,
                "legacy": False,
            },
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "target": "cheff.ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 3,
                "ckpt_path": ae_path,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        }


class CheffLDMTab2I_Imgonly(CheffLDMTab2IAdaptor):
    @torch.no_grad()
    def sample(
        self,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: dict = None,
        batch_size: int = 1,
        *args,
        **kwargs,
    ) -> Tensor:
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps, conditioning=conditioning, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion_tab:
        config_dict = self._get_config_dict(self, ae_path)
        model = LatentDiffusion_tab(**config_dict)

        state_dict = torch.load(model_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)
        return model

    @staticmethod
    def _get_config_dict(self, ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "",
            "cond_stage_key": "prev_img",
            "image_size": 64,
            "channels": 3,
            "cond_stage_trainable": False,
            "conditioning_key": "concat",
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "unet_config": CheffLDMTab2I_Imgonly._get_unet_config_dict(self),
            "first_stage_config": CheffLDMTab2I_Imgonly._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMTab2I_Imgonly._get_cond_config_dict(self),
        }

    @staticmethod
    def _get_cond_config_dict(self) -> Dict:
        return "__is_first_stage__"

    @staticmethod
    def _get_unet_config_dict(self) -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 6,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_heads": 8,
                "use_spatial_transformer": False,
                "transformer_depth": 1,
                "use_checkpoint": True,
                "legacy": False,
            },
        }
