import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BartConfig
from transformers.modeling_bart import (
    BartEncoder, BartForConditionalGeneration, PretrainedBartModel, LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding, SelfAttention, ACT2FN, LayerNorm, invert_mask, _make_linear_from_emb,
    _prepare_bart_decoder_inputs, _filter_out_falsey_values,
)

from .mem_utils import AttnAggregator, get_bsz, fill_dummy_memory, augment_decoder_cache_with_memory, merge_states

logger = logging.getLogger(__name__)


class MultipleKeyAttention(nn.Module):
    def __init__(
            self,
            n_input: int,
            embed_dim: int,
            num_heads,
            dropout=0.0,
            bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.n_input = n_input
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert self.n_input > 1
        self.extra_k_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=bias)
                                           for _ in range(1, self.n_input)])
        self.extra_v_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=bias)
                                           for _ in range(1, self.n_input)])
        self.extra_q_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=bias)
                                           for _ in range(1, self.n_input)])

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    @property
    def k_projs(self):
        return [self.k_proj, ] + list(self.extra_k_proj)

    @property
    def v_projs(self):
        return [self.v_proj, ] + list(self.extra_v_proj)

    @property
    def q_projs(self):
        return [self.q_proj, ] + list(self.extra_q_proj)

    def forward(
            self,
            query,
            keys: List[Tuple[Tensor, Optional[Tensor]]],
            layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
            need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        use_saved_states = False
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key_0" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                use_saved_states = True
        else:
            saved_state = None
            layer_state = {}

        assert self.n_input == len(keys)

        # KV calculation for each set of inputs
        if use_saved_states:
            k = [None for _ in range(self.n_input)]
            v = [None for _ in range(self.n_input)]
        else:
            k = []
            v = []
            for i in range(self.n_input):
                key = keys[i][0]
                k.append(self._shape(self.k_projs[i](key), -1, bsz))
                v.append(self._shape(self.v_projs[i](key), -1, bsz))

        # Read & save cache
        layer_state[self.cache_key] = layer_state.get(self.cache_key, {})
        for i in range(self.n_input):
            if saved_state is not None:
                k[i], v[i] = self._use_saved_state(i, k[i], v[i], saved_state, bsz)
            assert k[i] is not None
            layer_state[self.cache_key].update({
                f"prev_key_{i}": k[i].view(bsz, self.num_heads, -1, self.head_dim),
                f"prev_value_{i}": v[i].view(bsz, self.num_heads, -1, self.head_dim),
            })

        # KV padding mask for each set of inputs
        key_padding_mask = [x[1] for x in keys]

        # QKV & attention calculation for each set of inputs
        attn_weights_all = []
        for i in range(self.n_input):
            q = self.q_projs[i](query) * self.scaling

            q = self._shape(q, tgt_len, bsz)

            src_len = k[i].size(1)
            attn_weights = torch.bmm(q, k[i].transpose(1, 2))
            assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

            # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
            if key_padding_mask[i] is not None and key_padding_mask[i].dim() == 0:
                key_padding_mask[i] = None
            assert key_padding_mask[i] is None or key_padding_mask[i].size()[:2] == (bsz, src_len,)

            if key_padding_mask[i] is not None:  # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                reshaped = key_padding_mask[i].unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights_all.append(attn_weights)

        attn_weights_all = torch.cat(attn_weights_all, dim=-1)
        attn_weights_all = F.softmax(attn_weights_all, dim=-1)
        attn_probs = F.dropout(attn_weights_all, p=self.dropout, training=self.training)

        v = torch.cat(v, dim=1)
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights_all = attn_weights_all.view(bsz, self.num_heads, tgt_len, sum(kk.size(1) for kk in k))
        else:
            attn_weights_all = None
        return attn_output, attn_weights_all

    def _use_saved_state(self, ind: int, k, v, saved_state, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if f"prev_key_{ind}" in saved_state:
            _prev_key = saved_state[f"prev_key_{ind}"]
            assert _prev_key is not None
            k = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
        if f"prev_value_{ind}" in saved_state:
            _prev_value = saved_state[f"prev_value_{ind}"]
            assert _prev_value is not None
            v = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
        assert k is not None and v is not None
        return k, v


class DecoderLayerMultipleKey(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MultipleKeyAttention(
            2, self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            encoder_hidden_states,
            # encoder_attn_mask=None,
            layer_state=None,
            causal_mask=None,
            decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            keys=encoder_hidden_states,
            # key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartMemoryMultipleKeyDecoder(nn.Module):
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx,
            )
        self.layers = nn.ModuleList(
            [DecoderLayerMultipleKey(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            memory_states,
            encoder_padding_mask,
            decoder_padding_mask,
            decoder_causal_mask,
            decoder_cached_states=None,
            use_cache=False,
            **unused
    ):
        # fix memory states and padding mask
        memory_states, memory_attn_mask = merge_states(memory_states)

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)
        if memory_attn_mask is not None:
            memory_attn_mask = invert_mask(memory_attn_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        memory_states = memory_states.transpose(0, 1)
        keys = [(encoder_hidden_states, encoder_padding_mask), (memory_states, memory_attn_mask)]

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                keys,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class BartModelEMem(PretrainedBartModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartMemoryMultipleKeyDecoder(config, self.shared)

        self.init_weights()

        self.init_aggregator = AttnAggregator(config.d_model)
        self.update_aggregator = AttnAggregator(config.d_model)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly

    def forward(
            self,
            input_ids,
            memory_update_info=None,
            history_data=None,
            attention_mask=None,
            decoder_input_ids=None,
            labels_segment_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=False,
    ):
        if history_data is not None:
            assert encoder_outputs is None  # Just for checking. Logics are confusing
            memory_update_info = self.forward_history(history_data)

        bsz = get_bsz(input_ids, encoder_outputs)
        memory_update_info = fill_dummy_memory(memory_update_info, self.config, bsz)
        # NOTE: memory_update_info is only required when encoder_outputs is not present

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        assert isinstance(encoder_outputs, tuple)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            memory_update_info,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        decoder_outputs = augment_decoder_cache_with_memory(decoder_outputs, memory_update_info)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs

    def forward_history(self, history):
        ret = []
        for i, length in enumerate(history["length"]):
            memory_update_info = fill_dummy_memory(None, self.config, 1)
            for j in range(length):
                h = {k: history[k][i, j].unsqueeze(0) for k in history.keys()
                     if k not in ["length", "metadata", "lm_labels", ]}
                with torch.no_grad():
                    outs = self.forward(memory_update_info=memory_update_info, **h)
                memory_update_info = self.get_memory_update_info(h, outs, memory_update_info)
            ret += memory_update_info
        return ret

    def get_memory_update_info(self, input_kwargs, outs, memory_update_info):
        # Only called when not in generation mode
        decoder_outputs, encoder_outputs = outs
        labels_segment_ids = input_kwargs["labels_segment_ids"]

        memory_update_info = fill_dummy_memory(
            memory_update_info, self.config,
            get_bsz(input_kwargs.get("input_ids", None), input_kwargs.get("encoder_outputs", None))
        )

        memory_update_info_ret = []
        for i in range(len(memory_update_info)):
            mem_new = []

            # Update memory if entity states are updated
            for j in range(1, memory_update_info[i].shape[0] + 1):
                h = memory_update_info[i][j - 1]
                if (labels_segment_ids[i] == j).sum() > 0:
                    h_new = decoder_outputs[i][labels_segment_ids[i] == j]
                    h_new = self.update_aggregator(h_new)
                    h = (h + h_new) / 2
                mem_new.append(h)

            # Initialize memory by decoder embeddings
            for j in range(memory_update_info[i].shape[0] + 1, labels_segment_ids[i].max() + 1):
                assert (labels_segment_ids[i] == j).sum() > 0
                h_new = decoder_outputs[i][labels_segment_ids[i] == j]
                h_new = self.init_aggregator(h_new)
                mem_new.append(h_new)

            if len(mem_new) > 0:
                mem_new = torch.stack(mem_new, dim=0)
            else:
                mem_new = fill_dummy_memory(None, self.config, 1)[0]
            memory_update_info_ret.append(mem_new)

        return memory_update_info_ret


class BartForConditionalGenerationEMem(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__(config)
        base_model = BartModelEMem(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.unused_keys = []

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            lm_labels=None,
            use_cache=False,
            **unused
    ):
        history_data = dict()
        for k in list(unused.keys()):
            if k.startswith("history_"):
                history_data[k[len("history_"):]] = unused.pop(k)
        if len(history_data) == 0:
            history_data = None

        outputs = self.model(
            input_ids,
            history_data=history_data,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            **unused
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        past, memory_update_info = past
        ret = super(BartForConditionalGenerationEMem, self).prepare_inputs_for_generation(
            decoder_input_ids, past, attention_mask, use_cache, **kwargs
        )
        ret['memory_update_info'] = memory_update_info
        return ret

    def _reorder_cache(self, past, beam_idx):
        past, memory_update_info = past
        ret = super(BartForConditionalGenerationEMem, self)._reorder_cache(past, beam_idx)
        memory_update_info = [memory_update_info[i] for i in beam_idx]
        return ret, memory_update_info

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            memory_update_info=None,
            **model_specific_kwargs
    ) -> torch.LongTensor:
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
                isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
                isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
                bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                        num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                        num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
                self.config.is_encoder_decoder
                and hasattr(self.config, "decoder")
                and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                    decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            memory_update_info = fill_dummy_memory(memory_update_info, self.config, input_ids.shape[0])

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                    batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])
            memory_update_info = [memory_update_info[i] for i in expanded_batch_idxs]

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=(encoder_outputs, memory_update_info),
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=(encoder_outputs, memory_update_info),
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output  # , _filter_out_falsey_values(encoder_outputs)
