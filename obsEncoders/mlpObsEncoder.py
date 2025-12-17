
from .basePolicyModule import PolicyModule, PolicyOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from factories.mlpfac import MLP

box_obs = ['attachment_time_left', 'aux_ticks', 'center_path', 'center_path_distance', 'distance_down_track', 'energy', 'front', 'max_steer_angle', 'shield_time', 'skeed_factor', 'velocity']

seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']

discrete_action_keys = ['brake', 'drift', 'fire', 'nitro', 'rescue']
continuous_action_keys =  ['acceleration', 'steer']


class MLPboxObservationEncoder(nn.Module):
    
    """
        simple MLP encoder for discrete action spaces, ouput a latent vector for each action
    """
    def __init__(self, input_dim, config, output_dim, device):
        self.device = device
        self.encoder = MLP(input_dim, config, output_dim)

    def forward(self, x):
        return self.encoder(x.to(self.device))
        
        
class seqObservationEncoder(nn.Module):
    """
        implements transformer-based encoder for sequence observation for later contecanation for one type
        Encodes sequence observation of shape:
        seq_obs: (batch_size, T, N, obs_dim) where T is the number of type steps, N is the number of entities
        seq_obs_mask: (batch_size, T, N) binary mask indicating valid entities
    """
    def __init__(
        self, 
        device,
        raw_dim,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_projection = nn.Linear(raw_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            device = device,
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
            batch_first=True,
            norm_first=True,
        )
        self.device = device
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model) if use_layernorm else None,
        )
        self.attn_pool = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, seq_obs, seq_obs_mask):
        B, T, N, _ = seq_obs.shape

        x = seq_obs.view(B * T, N, -1)
        mask = ~seq_obs_mask.view(B * T, N).bool()

        x = self.input_projection(x)
        x = self.transformer(x.to(self.device), src_key_padding_mask=mask.to(self.device))

        attn_logits = self.attn_pool(x.to(self.device)).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)

        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        embeddings = pooled.view(B, T, self.d_model)

        return embeddings

class MLPObsEncoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()


        self.device = device
        self.box_obs = ['attachment_time_left', 'aux_ticks', 'center_path', 'center_path_distance', 'distance_down_track', 'energy', 'front', 'max_steer_angle', 'shield_time', 'skeed_factor', 'velocity']

        self.seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']
        box_input_dim = len(box_obs)
        box_output_dim = config['box_encoder']['output_dim']
        self.boxEncoder = MLPboxObservationEncoder(box_input_dim, config['box_encoder'], box_output_dim, self.device)


        self.seqEncoder = nn.ModuleDict()
        for obs_key in self.seq_obs:
            feature_config = config['seq_encoder'][obs_key]
            self.seqEncoder[obs_key] = seqObservationEncoder(
                self.device, **feature_config
            )
        self.output_dim = box_output_dim + d_model

    def forward(self, Batch):
        
        box_obs = torch.cat(Batch.box_obs, dim=-1)
        seq_obs = Batch.seq_obs
        seq_obs_mask = Batch.seq_obs_mask

        box_embedding = self.boxEncoder(box_obs)
        seq_embeddings = {
            k: self.seqEncoder[k](v, seq_obs_mask[k])
            for k, v in self.seq_obs.items()
        }
        return torch.cat([box_embedding] + list(seq_embeddings.values())], dim=-1)


