"""Inkling-2 attention a block of queries at a time.

Inkling-2's attention is eager-only in transformers 5.24: a hidden-state
conditioned relative-position bias and one sink logit per query rule out SDPA
and flash attention. Both the scores and the bias are materialised as
(heads x queries x keys) tables, so memory grows with the square of the
sequence (an fp32 table is 1.5 GiB at 4k tokens, 24 GiB at 16k).

Each query row is independent (its own softmax over all keys plus its sink),
so the same computation over blocks of queries gives the same result. The
bias is built lazily, only for a block's rows. In training each block is
checkpointed: backward keeps only the block's inputs and rebuilds its tables,
otherwise autograd would save every block's tables and be quadratic again.
"""

import torch
from torch.utils.checkpoint import checkpoint

NAME = "songe_blocks"
# Elements per block's score table; fp32 softmax plus its inputs stays near
# 1.5 GB whatever the length.
BLOCK_ELEMENTS = 2**27


class LazyBias:
    """A relative-position bias whose rows are computed on request."""

    def __init__(self, module, hidden_states, relative_distance, position_ids):
        self.module, self.hidden_states = module, hidden_states
        self.relative_distance, self.position_ids = relative_distance, position_ids

    def rows(self, start, end, hidden_states=None):
        hidden = self.hidden_states[:, start:end] if hidden_states is None else hidden_states
        return self.module.dense_forward(
            hidden, self.relative_distance[start:end], self.position_ids[..., start:end]
        )

    def dense(self):
        return self.rows(0, self.hidden_states.shape[1])


def install(model):
    """Switch every Inkling-2 attention layer in model to block attention."""
    from transformers import AttentionInterface
    from transformers.models.inkling2_loop import modeling_inkling2_loop as m

    bias_class = m.Inkling2LoopRelativePositionBias
    if not hasattr(bias_class, "dense_forward"):
        bias_class.dense_forward = bias_class.forward

        def lazy_forward(self, hidden_states, relative_distance, position_ids):
            if getattr(self, "songe_lazy", False):
                return LazyBias(self, hidden_states, relative_distance, position_ids)
            return self.dense_forward(hidden_states, relative_distance, position_ids)

        bias_class.forward = lazy_forward
    eager = m.eager_attention_forward

    def blocks(module, query, key, value, attention_mask, scaling, dropout=0.0,
               position_bias=None, sinks=None, **kwargs):
        queries, keys = query.shape[2], key.shape[2]
        heads = query.shape[1]
        size = max(64, min(1024, BLOCK_ELEMENTS // max(1, heads * keys)))
        if not isinstance(position_bias, LazyBias):
            # Not built by an Inkling attention layer: the original path.
            return eager(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                         position_bias=position_bias, sinks=sinks, **kwargs)
        if queries <= size:
            return eager(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                         position_bias=position_bias.dense(), sinks=sinks, **kwargs)

        def block(q, k, v, mask, sink, hidden, start, end):
            bias = position_bias.rows(start, end, hidden)
            out, _ = eager(module, q, k, v, mask, scaling=scaling, dropout=dropout,
                           position_bias=bias, sinks=sink, **kwargs)
            return out

        training = torch.is_grad_enabled() and module.training
        outputs = []
        for start in range(0, queries, size):
            end = min(start + size, queries)
            args = (
                query[:, :, start:end],
                key,
                value,
                None if attention_mask is None else attention_mask[..., start:end, :],
                None if sinks is None else sinks[:, :, start:end],
                position_bias.hidden_states[:, start:end],
                start,
                end,
            )
            outputs.append(checkpoint(block, *args, use_reentrant=False) if training else block(*args))
        # eager returns (batch, queries, heads, dim): blocks join on queries.
        return torch.cat(outputs, dim=1), None

    AttentionInterface.register(NAME, blocks)
    # transformers builds masks only for implementations it knows; without
    # this the blocks would see no causal mask at all. Same masks as eager.
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    for module in model.modules():
        if isinstance(module, bias_class):
            module.songe_lazy = True
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config._attn_implementation = NAME
    for module in model.modules():
        if hasattr(module, "config") and getattr(module.config, "model_type", "").startswith("inkling2"):
            module.config._attn_implementation = NAME
