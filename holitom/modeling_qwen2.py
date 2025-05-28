from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding, logger

class Qwen2Model_holitom(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        import os
        HOLITOM_k = int(os.getenv("HOLITOM_k", 3))
        HOLITOM_r = float(os.getenv("HOLITOM_r", 0.5))
        HOLITOM_image_token_start_index = self.image_token_posi[0]
        HOLITOM_image_token_length = self.image_tokens[0]
        seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1]

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if layer_idx < HOLITOM_k:
                    pruned_attention_mask = None
                elif layer_idx == HOLITOM_k and position_ids.size(1) > 1:
                    # compute pruned tokens, generate fastv sign
                    last_layer_attention = layer_outputs[1]
                    # compute average attention over different head
                    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                    # generate new attention mask based on the average attention, sample the top ATTENTION_RANK tokens with highest attention
                    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                    # get the attention in image token
                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
                        HOLITOM_image_token_start_index:HOLITOM_image_token_start_index+HOLITOM_image_token_length
                    ]
                    # get the indexs of the top ATTENTION_RANK tokens
                    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(
                        round(HOLITOM_image_token_length*(1-HOLITOM_r))
                    ).indices + HOLITOM_image_token_start_index
                    print("Before merge:", HOLITOM_image_token_length, "After merge:", round(HOLITOM_image_token_length*(1-HOLITOM_r)))
                    
                    
                    device = hidden_states.device
                    # [modified]
                    all_indices = torch.arange(HOLITOM_image_token_length, device=device)
                    non_topk_mask = ~torch.isin(all_indices, top_attention_rank_index - HOLITOM_image_token_start_index)
                    non_topk_indices = all_indices[non_topk_mask] + HOLITOM_image_token_start_index
                    non_topk_states = hidden_states[:, non_topk_indices, :]  # [batch_size, len(non_topk), hidden_size]
                    topk_states = hidden_states[:, top_attention_rank_index, :]  # [batch_size, len(topk), hidden_size]
                    non_topk_norm = torch.norm(non_topk_states, dim=-1, keepdim=True)  # [batch_size, len(non_topk), 1]
                    topk_norm = torch.norm(topk_states, dim=-1, keepdim=True)  # [batch_size, len(topk), 1]
                    dot_product = torch.bmm(non_topk_states, topk_states.transpose(1, 2))  # [batch_size, len(non_topk), len(topk)]
                    sim_matrix = dot_product / (non_topk_norm * topk_norm.transpose(1, 2))
                    sim_max, sim_max_index = torch.max(sim_matrix, dim=-1)
                    
                    for b in range(hidden_states.size(0)):
                        for i in range(len(non_topk_indices)):
                            non_topk_idx = non_topk_indices[i]
                            most_similar_topk_idx = top_attention_rank_index[sim_max_index[b, i]]
                            hidden_states[b, most_similar_topk_idx, :] = (hidden_states[b, most_similar_topk_idx, :] + hidden_states[b, non_topk_idx, :]) / 2
                    # [modified]
                    
                    # keep index
                    keep_indexs = torch.cat(
                        (
                            torch.arange(HOLITOM_image_token_start_index,device=device), 
                            top_attention_rank_index, 
                            torch.arange(
                                HOLITOM_image_token_start_index+HOLITOM_image_token_length,
                                seq_length_with_past,
                                device=device
                            )
                        )
                    )
                    # sort index
                    keep_indexs = keep_indexs.sort().values
                    # update seq length
                    new_seq_length = keep_indexs.shape[0]
                    # filter hidden states
                
                    hidden_states = hidden_states[:,keep_indexs,:] # lead the cuda error in the second iteration of decoding layeridx 3
                    # update position ids
                    position_ids = keep_indexs.unsqueeze(0)
                    
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    
                    cache_position = cache_position[:new_seq_length]   
                
                if layer_idx == HOLITOM_k - 1:
                    output_attentions = True
                else:
                    output_attentions = False
                    
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

