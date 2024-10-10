import copy
import pickle

import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, src):
        q = k = v = tgt

        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        q = tgt
        k = v = src

        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[
            0
        ].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, src):
        output = tgt

        for _, layer in enumerate(self.layers):
            output = layer(output, src)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_dim,
        dim,
        depth,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.z_dim = dim
        self.dropout = nn.Dropout(emb_dropout)
        self.to_cls_token = nn.Identity()
        self.deep = {}
        self.res = {}
        self.depth = depth
        for i in range(self.depth):
            if i == 0:
                input_dim = patch_dim
            else:
                input_dim = self.z_dim

            self._modules["depth_{}".format(i)] = nn.Sequential(
                nn.Linear(input_dim, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 4),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 4),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 4, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 1),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 1),
                nn.Dropout(dropout),
            )
            self._modules["encode_{}".format(i)] = nn.TransformerEncoderLayer(
                d_model=self.z_dim, nhead=2
            )
            self._modules["trans_{}".format(i)] = nn.TransformerEncoder(
                self._modules["encode_{}".format(i)], num_layers=1
            )
            self._modules["res_{}".format(i)] = nn.Linear(patch_dim, self.z_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.z_dim * 1), nn.Linear(self.z_dim * 1, num_classes)
        )


    def forward(self, samples: NestedTensor, couple=False):
        if couple:
            img = samples
        else:
            img, mask = samples.tensors, samples.mask

        x = img.clone()

        for i in range(self.depth):
            x = self._modules["depth_{}".format(i)](x)

            x = self._modules["trans_{}".format(i)](x.transpose(0, 1))
            x = x.transpose(0, 1)
            x = x + torch.relu(self._modules["res_{}".format(i)](img))
        return (self.mlp_head(x), x)


class WSIAggregator(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False
    ):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_heads=nhead
        )

        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )

    def forward(self, tgt, memory):
        hs = self.decoder(tgt, memory)
        return hs


class DeepGEM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        patch_dim=768,
        dim=32,
        depth=6,
        decoder_dropout=0.1,
        num_queries=5,
    ):
        super().__init__()
        self.patch_encoder = TransformerBackbone(
            num_classes,
            patch_dim,
            dim,
            depth,
            dropout=0.0,
            emb_dropout=0.0,
        )
        self.wsi_aggregator = WSIAggregator(
            d_model=dim,
            nhead=4,
            num_decoder_layers=2,
            dim_feedforward=dim * 2,
            dropout=decoder_dropout,
            activation="relu",
            return_intermediate_dec=False
        )

        self.query_embed = nn.Embedding(num_queries, dim)
        self.wsi_classifier = nn.Linear(dim * num_queries, num_classes)
        self.register_buffer("prototypes", torch.zeros(num_classes, dim))

    def forward(self, samples: NestedTensor, couple=False, partial_Y=None, args=None):
        if couple:
            img = samples
        else:
            img, mask = samples.tensors, samples.mask
        bs, patch_num, _ = img.shape

        patch_classifier_output, x_instance = self.patch_encoder(samples, couple)
        x_instance = x_instance.reshape(bs * patch_num, -1)
        patch_classifier_output = patch_classifier_output.reshape(bs * patch_num, -1)

        # compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(x_instance.detach(), prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        x_instance = x_instance.reshape(bs, patch_num, -1)
        patch_classifier_output = patch_classifier_output.reshape(bs, patch_num, -1)

        tgt = self.query_embed.weight
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        hs = self.wsi_aggregator(tgt, x_instance)
        hs = hs.view(hs.shape[0], -1)
        wsi_classifier_output = self.wsi_classifier(hs)

        return wsi_classifier_output



def build(args):
    device = torch.device("cuda")

    with open(args.checkpoint, "rb") as f:
        parameter = pickle.load(f)["parameter"]
    hidden_dim = parameter['hidden_dim']
    depth = parameter['depth']

    model = DeepGEM(
        num_classes=2,
        patch_dim=768,
        dim=hidden_dim,
        depth=depth,
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    return model, criterion
