import os
import re
import math
import torch
import importlib
from torch import nn
from einops import rearrange
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from cheff.ldm.models.autoencoder import AutoencoderKL
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from cheff.ldm.modules.x_transformer import (
    Encoder,
    AbsolutePositionalEmbedding,
    TransformerWrapper,
)  
from cheff.ldm.modules.clip import LayerNorm, Transformer
from cheff.ldm.util import instantiate_from_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class MultiClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Sequential(
            nn.Linear(in_features=n_classes, out_features=embed_dim),
            nn.GELU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.GELU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
        )

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        c = batch[key]
        c = self.embedding(c)
        c = c.unsqueeze(1)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=150, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size, max_seq_len=max_seq_len, attn_layers=Encoder(dim=n_embed, depth=n_layer)
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=150):
        super().__init__()
        from transformers import BertTokenizerFast

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self, n_embed, n_layer, vocab_size=30522, max_seq_len=150, device="cuda", use_tokenizer=True, embedding_dropout=0.0
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len, device=device)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class LineartabCLIPEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed=1536,
        max_seq_len=1024,
        context_embed=1280,
        feature_dim=768,
        checkpoint=None,
        use_pos_emb=True,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.encoder = nn.Linear(n_embed, context_embed)
        self.pos_emb = AbsolutePositionalEmbedding(n_embed, max_seq_len) if use_pos_emb else always(0)
        self.feature_extractor = nn.Sequential(nn.ReLU(), nn.Linear(context_embed, feature_dim))

        if checkpoint:
            print(f"Initialize tab embedder enc: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)["state_dict"]
            state_dict = {k.replace("model.transformer.", ""): v for k, v in state_dict.items() if "model.transformer" in k}
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            try:
                assert len(missing) == 0 and len(unexpected) == 0
            except:
                print("missing: ", missing)
                print("unexpected: ", unexpected)
            del state_dict

    def forward(self, tab_emb):
        tab_emb = tab_emb + self.pos_emb(tab_emb)
        tab_emb = tab_emb.float()
        z = self.encoder(tab_emb)
        z = self.feature_extractor(z.mean(dim=1))
        return z

    def encode(self, tab):
        return self(tab)


class TabTransformerEmbedder(nn.Module):
    def __init__(self, n_embed=1536, n_head=24, feature_dim=768, return_avg_embedding=False):
        super().__init__()
        self.return_avg_embedding = return_avg_embedding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embed, nhead=n_head, dim_feedforward=n_embed * 4, batch_first=True), 2
        )
        self.fc = nn.Linear(in_features=n_embed, out_features=feature_dim)

    def forward(self, emb, attn_mask):
        x = self.transformer(emb, src_key_padding_mask=attn_mask)
        x = self.fc(x)
        if self.return_avg_embedding:
            x = x.sum(1) / (~attn_mask).float().sum(dim=1, keepdim=True)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class TabAdaptor(AbstractEncoder):
    # clip tab + vae
    def __init__(
        self,
        autoencoder_config,
        clip_enc_checkpoint,
        context_dim,
        condition_feat_dim=512,
        device="cuda",
        clip_trainable=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.clip_emb = 768
        self.clip_tab_encoder = TabTransformerEmbedder(n_embed=1536, n_head=24, feature_dim=768, return_avg_embedding=False)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)["state_dict"]
            state_dict = {k.replace("model.transformer.", ""): v for k, v in state_dict.items() if "model.transformer" in k}
            missing, unexpected = self.clip_tab_encoder.load_state_dict(state_dict, strict=False)
            try:
                assert len(missing) == 0 and len(unexpected) == 0
            except:
                print("missing: ", missing)
                print("unexpected: ", unexpected)
            if not clip_trainable:
                self.clip_tab_encoder.train = disabled_train
                for param in self.clip_tab_encoder.parameters():
                    param.requires_grad = False
            del state_dict

        self.autoencoder = AutoencoderKL(**autoencoder_config)  # 3, 64, 64
        self.autoencoder = self.autoencoder.eval()
        self.autoencoder.train = disabled_train
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.vae_projector = nn.Linear(64 * 64, self.clip_emb)
        self.adaptor = nn.Linear(1024 + 3, condition_feat_dim)  # # of table token + vae feature channel
        if context_dim != self.clip_emb:
            self.projector = nn.Linear(self.clip_emb, context_dim)
        else:
            self.projector = None

    def forward(self, data):
        img = data["prev_img"]
        tab = data["table"]
        attn_mask = data["attn_mask"]

        # Encode table data
        clip_tab_emb = self.clip_tab_encoder(tab, attn_mask)

        # Encode image data
        vae_emb = self.autoencoder.encode(img).mode().detach()
        vae_emb = rearrange(vae_emb, "b c h w -> b c (h w)")
        vae_emb = self.vae_projector(vae_emb)

        # Fusion
        emb = torch.cat([clip_tab_emb, vae_emb], dim=1)
        emb = rearrange(emb, "b c d -> b d c")
        emb = self.adaptor(emb)
        emb = rearrange(emb, "b d c -> b c d")

        # project for Unet embedding
        if self.projector:
            emb = self.projector(emb)

        return emb

    def encode(self, data):
        return self(data)


class VAEAdaptor(AbstractEncoder):
    # VAE only
    def __init__(self, autoencoder_config, context_dim, device="cuda", **kwargs):
        super().__init__()
        self.device = device
        self.autoencoder = AutoencoderKL(**autoencoder_config)  # 3, 64, 64
        self.autoencoder = self.autoencoder.eval()
        self.autoencoder.train = disabled_train
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.vae_projector = nn.Linear(64 * 64, context_dim)

    def forward(self, img):
        vae_emb = self.autoencoder.encode(img).mode().detach()
        vae_emb = rearrange(vae_emb, "b c h w -> b c (h w)")
        vae_emb = self.vae_projector(vae_emb)
        return vae_emb

    def encode(self, img):
        return self(img)


class ImgAdaptor(AbstractEncoder):
    # Clip img + VAE
    def __init__(
        self,
        clip_visual_enc_config,
        clip_enc_checkpoint,
        autoencoder_config,
        context_dim,
        condition_feat_dim=512,
        device="cuda",
        clip_trainable=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.clip_emb = 768
        self.clip_img_encoder = VisualTransformer(**clip_visual_enc_config)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)["state_dict"]
            img_state_dict = {k.replace("model.visual.", ""): v for k, v in state_dict.items() if "model.visual" in k}
            self.init_checkpoint("clip_img_encoder", img_state_dict, trainable=clip_trainable)
            del state_dict

        self.autoencoder = AutoencoderKL(**autoencoder_config)  # 3, 64, 64
        self.autoencoder = self.autoencoder.eval()
        self.autoencoder.train = disabled_train
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.vae_projector = nn.Linear(64 * 64, self.clip_emb)
        self.adaptor = nn.Linear(3 + 65, condition_feat_dim)  # # of table token + vae feature channel + vit token len
        if context_dim != self.clip_emb:
            self.projector = nn.Linear(self.clip_emb, context_dim)
        else:
            self.projector = None

    def init_checkpoint(self, model_name, state_dict, trainable=False):
        print(f"Load {model_name} checkpoint...")
        missing, unexpected = getattr(self, model_name).load_state_dict(state_dict, strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)

        if not trainable:
            getattr(self, model_name).train = disabled_train
            for param in getattr(self, model_name).parameters():
                param.requires_grad = False

    def forward(self, img):
        # Encode image data
        clip_img_emb = self.clip_img_encoder(img)  # [16, 65, 768]
        vae_emb = self.autoencoder.encode(img).mode().detach()  # [b, 3, 64, 64]
        vae_emb = rearrange(vae_emb, "b c h w -> b c (h w)")
        vae_emb = self.vae_projector(vae_emb)  # [b, 3, 768]

        # Fusion
        emb = torch.cat([clip_img_emb, vae_emb], dim=1)  # [b, 65 + 3, 768]
        emb = rearrange(emb, "b c d -> b d c")
        emb = self.adaptor(emb)
        emb = rearrange(emb, "b d c -> b c d")  # [b, feat_dim, 768]

        # project for Unet embedding
        if self.projector:
            emb = self.projector(emb)

        return emb

    def encode(self, img):
        return self(img)
        

class ClipTabAdaptor(AbstractEncoder):
    # Clip tab
    def __init__(
        self,
        clip_enc_checkpoint,
        context_dim,
        device="cuda",
        clip_trainable=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.clip_emb = 768
        self.clip_tab_encoder = TabTransformerEmbedder(n_embed=1536, n_head=24, feature_dim=768, return_avg_embedding=False)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)["state_dict"]
            tab_state_dict = {k.replace("model.transformer.", ""): v for k, v in state_dict.items() if "model.transformer" in k}
            self.init_checkpoint("clip_tab_encoder", tab_state_dict, trainable=clip_trainable)
            del state_dict

        if context_dim != self.clip_emb:
            self.projector = nn.Linear(self.clip_emb, context_dim)
        else:
            self.projector = None

    def init_checkpoint(self, model_name, state_dict, trainable=False):
        print(f"Load {model_name} checkpoint...")
        missing, unexpected = getattr(self, model_name).load_state_dict(state_dict, strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)

        if not trainable:
            getattr(self, model_name).train = disabled_train
            for param in getattr(self, model_name).parameters():
                param.requires_grad = False

    def forward(self, data):
        tab = data["table"]
        attn_mask = data["attn_mask"]

        # Encode table data
        emb = self.clip_tab_encoder(tab, attn_mask)  # [16, 1024, 768]

        # project for Unet embedding
        if self.projector:
            emb = self.projector(emb)

        return emb

    def encode(self, data):
        return self(data)


class ClipImgAdaptor(AbstractEncoder):
    # Clip img
    def __init__(
        self,
        clip_visual_enc_config,
        clip_enc_checkpoint,
        context_dim,
        device="cuda",
        clip_trainable=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.clip_emb = 768
        self.clip_img_encoder = VisualTransformer(**clip_visual_enc_config)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)["state_dict"]
            img_state_dict = {k.replace("model.visual.", ""): v for k, v in state_dict.items() if "model.visual" in k}
            self.init_checkpoint("clip_img_encoder", img_state_dict, trainable=clip_trainable)
            del state_dict

        if context_dim != self.clip_emb:
            self.projector = nn.Linear(self.clip_emb, context_dim)
        else:
            self.projector = None

    def init_checkpoint(self, model_name, state_dict, trainable=False):
        print(f"Load {model_name} checkpoint...")
        missing, unexpected = getattr(self, model_name).load_state_dict(state_dict, strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)

        if not trainable:
            getattr(self, model_name).train = disabled_train
            for param in getattr(self, model_name).parameters():
                param.requires_grad = False

    def forward(self, img):
        emb = self.clip_img_encoder(img)  # [16, 65, 768]

        # project for Unet embedding
        if self.projector:
            emb = self.projector(emb)

        return emb

    def encode(self, data):
        return self(data)


class ClipAdaptor(AbstractEncoder):
    # Clip tab & Clip img
    def __init__(
        self,
        clip_visual_enc_config,
        clip_enc_checkpoint,
        context_dim,
        condition_feat_dim=512,
        device="cuda",
        clip_trainable=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.clip_emb = 768
        self.clip_img_encoder = VisualTransformer(**clip_visual_enc_config)
        self.clip_tab_encoder = TabTransformerEmbedder(n_embed=1536, n_head=24, feature_dim=768, return_avg_embedding=False)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)["state_dict"]
            img_state_dict = {k.replace("model.visual.", ""): v for k, v in state_dict.items() if "model.visual" in k}
            tab_state_dict = {k.replace("model.transformer.", ""): v for k, v in state_dict.items() if "model.transformer" in k}
            self.init_checkpoint("clip_img_encoder", img_state_dict, trainable=clip_trainable)
            self.init_checkpoint("clip_tab_encoder", tab_state_dict, trainable=clip_trainable)
            del state_dict

        self.adaptor = nn.Linear(1024 + 65, condition_feat_dim)  # # of table token + vae feature channel + vit token len
        if context_dim != self.clip_emb:
            self.projector = nn.Linear(self.clip_emb, context_dim)
        else:
            self.projector = None

    def init_checkpoint(self, model_name, state_dict, trainable=False):
        print(f"Load {model_name} checkpoint...")
        missing, unexpected = getattr(self, model_name).load_state_dict(state_dict, strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)

        if not trainable:
            getattr(self, model_name).train = disabled_train
            for param in getattr(self, model_name).parameters():
                param.requires_grad = False

    def forward(self, data):
        img = data["prev_img"]
        tab = data["table"]
        attn_mask = data["attn_mask"]

        # Encode events
        clip_tab_emb = self.clip_tab_encoder(tab, attn_mask)  # [16, 1024, 768]
        clip_img_emb = self.clip_img_encoder(img)  # [16, 65, 768]

        # Fusion
        emb = torch.cat([clip_tab_emb, clip_img_emb], dim=1)  # [b, 1024 + 65, 768]
        emb = rearrange(emb, "b c d -> b d c")
        emb = self.adaptor(emb)
        emb = rearrange(emb, "b d c -> b c d")  # [b, feat_dim, 768]

        # project for Unet embedding
        if self.projector:
            emb = self.projector(emb)

        return emb

    def encode(self, img):
        return self(img)


class MultiModalTransformerAdaptor(AbstractEncoder):
    # Clip tab & img & VAE
    def __init__(
        self,
        autoencoder_config,
        clip_visual_enc_config,
        clip_enc_checkpoint,
        context_dim,
        max_event_len=1024,
        condition_feat_dim=512,
        device="cuda",
        clip_trainable=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.clip_emb = 768
        self.max_event_len = max_event_len
        self.clip_img_encoder = VisualTransformer(**clip_visual_enc_config)
        self.clip_tab_encoder = TabTransformerEmbedder(n_embed=1536, n_head=24, feature_dim=768, return_avg_embedding=False)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            print()
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)["state_dict"]
            img_state_dict = {k.replace("model.visual.", ""): v for k, v in state_dict.items() if "model.visual" in k}
            tab_state_dict = {k.replace("model.transformer.", ""): v for k, v in state_dict.items() if "model.transformer" in k}
            self.init_checkpoint("clip_img_encoder", img_state_dict, trainable=clip_trainable)
            self.init_checkpoint("clip_tab_encoder", tab_state_dict, trainable=clip_trainable)
            del state_dict
        else:
            import clip
            print(f"Initialize clip enc (only img enc)")
            image_encoder, _ = clip.load("ViT-B/32", jit=False)
            state_dict = image_encoder.visual.state_dict()
            img_state_dict = {k:v for k, v in state_dict.items() if "positional_embedding" not in k}
            self.init_checkpoint("clip_img_encoder", img_state_dict, trainable=clip_trainable)
            del image_encoder, state_dict

        self.autoencoder = AutoencoderKL(**autoencoder_config)  # 3, 64, 64
        self.autoencoder = self.autoencoder.eval()
        self.autoencoder.train = disabled_train
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.vae_projector = nn.Linear(64 * 64, self.clip_emb)
        self.adaptor = nn.Linear(self.max_event_len + 3 + 65, condition_feat_dim)  # # of table token + vae feature channel + vit token len
        if context_dim != self.clip_emb:
            self.projector = nn.Linear(self.clip_emb, context_dim)
        else:
            self.projector = None

    def init_checkpoint(self, model_name, state_dict, trainable=False):
        print(f"Load {model_name} checkpoint...")
        missing, unexpected = getattr(self, model_name).load_state_dict(state_dict, strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)

        if not trainable:
            getattr(self, model_name).train = disabled_train
            for param in getattr(self, model_name).parameters():
                param.requires_grad = False

    def forward(self, data):
        img = data["prev_img"]
        tab = data["table"]
        attn_mask = data["attn_mask"]

        # Encode table data
        clip_tab_emb = self.clip_tab_encoder(tab, attn_mask)  # [16, 1024, 768]

        # Encode image data
        clip_img_emb = self.clip_img_encoder(img)  # [16, 65, 768]
        vae_emb = self.autoencoder.encode(img).mode().detach()  # [b, 3, 64, 64]
        vae_emb = rearrange(vae_emb, "b c h w -> b c (h w)")
        vae_emb = self.vae_projector(vae_emb)  # [b, 3, 768]

        # Fusion
        emb = torch.cat([clip_tab_emb, clip_img_emb, vae_emb], dim=1)  # [b, 1024 + 65 + 3, 768]
        emb = rearrange(emb, "b c d -> b d c")
        emb = self.adaptor(emb)
        emb = rearrange(emb, "b d c -> b c d")  # [b, feat_dim, 768]

        # project for Unet embedding
        if self.projector:
            emb = self.projector(emb)

        return emb

    def encode(self, data):
        return self(data)


class MultiModalAdaptor(AbstractEncoder):
    def __init__(self, autoencoder_config, tabencoder_config, context_dim, clip_enc_checkpoint, device="cuda"):
        super().__init__()
        self.device = device
        self.clip_encoder = resnet50()
        self.clip_encoder.fc = torch.nn.Linear(2048, 768)

        if clip_enc_checkpoint:
            print(f"Initialize clip enc: {clip_enc_checkpoint}")
            state_dict = torch.load(clip_enc_checkpoint, map_location=self.device)
            state_dict = {k.replace("model.visual.", ""): v for k, v in state_dict["state_dict"].items() if "model.visual" in k}
            missing, unexpected = self.clip_encoder.load_state_dict(state_dict, strict=False)
            try:
                assert len(missing) == 0 and len(unexpected) == 0
            except:
                print("missing: ", missing)
                print("unexpected: ", unexpected)
            del state_dict

        self.autoencoder = AutoencoderKL(**autoencoder_config)  # 3, 64, 64
        self.autoencoder = self.autoencoder.eval()
        self.autoencoder.train = disabled_train
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.projector = nn.Sequential(nn.Flatten(), nn.ReLU(), nn.Linear(12288, 768))
        self.tab_encoder = instantiate_from_config(tabencoder_config)
        self.adaptor = nn.Linear(768 * 3, context_dim)

    def forward(self, data):
        img = data["prev_img"]
        tab = data["table"]

        # image embedding
        clip_emb = self.clip_encoder(img)
        vae_emb = self.autoencoder.encode(img).mode().detach()
        vae_emb = self.projector(vae_emb)

        # table embedding
        tab_emb = self.tab_encoder(tab)

        # aggregator
        emb = torch.cat([clip_emb, vae_emb, tab_emb], dim=1)
        emb = self.adaptor(emb).unsqueeze(1)

        return emb

    def encode(self, data):
        return self(data)

