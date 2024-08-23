# coding=utf-8
"""PyTorch MAMBA model."""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from torch import nn
from torch.nn.parameter import Parameter
from transformers import MambaConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.models.mamba_cache import MambaCacheManager
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.worker.model_runner import (_BATCH_SIZES_TO_CAPTURE,
                                      _get_graph_batch_size)

KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class MambaCacheParams:
    is_prompt: bool = False
    conv_state: torch.Tensor = torch.Tensor()
    ssm_state: torch.Tensor = torch.Tensor()


# Load weights that are sharded along axis 0
def sharded_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
    tp_rank = get_tensor_model_parallel_rank()
    param.data.copy_(
        loaded_weight.data.split(param.data.shape[0], dim=0)[tp_rank])


class MambaRMSNormGated(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader})

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(
                gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx):
        super().__init__()
        self.config = config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_heads
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.use_conv_bias = config.use_conv_bias

        self.norm_before_gate = config.norm_before_gate
        self.layer_norm_epsilon = config.layer_norm_epsilon

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit

        self.use_bias = config.use_bias

        self.conv_dim = (self.intermediate_size +
                         2 * self.n_groups * self.ssm_state_size)
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=self.use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            bias=self.use_bias)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads // self.tp_size))

        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(self.time_step_rank,
                                            self.intermediate_size,
                                            bias=True,
                                            skip_bias_add=True)

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            sharded_weight_loader(param, -torch.exp(loaded_weight.float()))

        self.A = nn.Parameter(torch.ones(self.num_heads // self.tp_size))
        self.D = nn.Parameter(torch.ones(self.num_heads // self.tp_size))
        self.norm = MambaRMSNormGated(self.intermediate_size // self.tp_size,
                                      eps=self.layer_norm_epsilon)

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
        )
        self.activation = config.hidden_act

    def mamba_forward(self,
                      hidden_states: torch.Tensor,
                      cache_params: MambaCacheParams = None):
        # set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = (2 * self.intermediate_size + 2 * self.n_groups *
                       self.ssm_state_size + self.num_heads) // self.tp_size

        if cache_params is not None and not cache_params.is_prompt:
            in_projected_states, _ = self.in_proj(hidden_states.squeeze(1))
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [
                d_mlp, d_mlp, self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size, self.num_heads // self.tp_size
            ]
            _, _, gate, hidden_states_B_C, dt = torch.split(
                in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B_C = torch.split(
                hidden_states_B_C,
                [
                    self.intermediate_size // self.tp_size,
                    2 * groups_time_state_size // self.tp_size
                ],
                dim=-1,
            )

            B_C = tensor_model_parallel_all_gather(B_C.contiguous())
            B, C = torch.split(
                B_C, [groups_time_state_size, groups_time_state_size], dim=-1)

            A = self.A[:, None, ...][:, :, None].expand(
                -1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(
                batch_size, self.num_heads // self.tp_size, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_state,
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(
                batch_size, self.num_heads // self.tp_size * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[0][:, None, ...]
        # if no cache is found, calling the kernel
        else:
            #TODO: Handle this if needed
            # if (attention_mask is not None
            #     and attention_mask.shape[1] > 1
            #     and attention_mask.shape[0] > 1):
            #
            # # tune out hidden states for pad tokens,
            # # see https://github.com/state-spaces/mamba/issues/66
            #     dtype = hidden_states.dtype
            #     hidden_states = (hidden_states *
            #                      attention_mask[:, :, None]).to(dtype)

            # 1. Gated MLP's linear projection
            projected_states, _ = self.in_proj(hidden_states)
            dt_limit_kwargs = {} if self.time_step_limit == (
                0.0, float("inf")) else {
                    "dt_limit": self.time_step_limit
                }

            gate, hidden_states_B_C, time_step = torch.split(
                projected_states,
                [
                    self.intermediate_size // self.tp_size, self.conv_dim //
                    self.tp_size, self.num_heads // self.tp_size
                ],
                dim=-1,
            )

            time_step = nn.functional.softplus(time_step + self.dt_bias)
            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in [
                    "silu", "swish"
            ]:
                hidden_states_B_C = self.act(
                    self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(
                        1, 2)[:, :seq_len]
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
            else:
                hidden_states_B_C = causal_conv1d_fn(
                    x=hidden_states_B_C.transpose(1, 2),
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)[:, :seq_len]

            hidden_states, B_C = torch.split(
                hidden_states_B_C,
                [
                    self.intermediate_size // self.tp_size,
                    2 * groups_time_state_size // self.tp_size
                ],
                dim=-1,
            )

            # Allgather on B and C needed
            B_C = tensor_model_parallel_all_gather(B_C.contiguous())
            B, C = torch.split(
                B_C, [groups_time_state_size, groups_time_state_size], dim=-1)

            # if (attention_mask is not None
            #     and attention_mask.shape[1] > 1
            #     and attention_mask.shape[0] > 1:
            #
            #     # tune out hidden states for pad tokens,
            #     # see https://github.com/state-spaces/mamba/issues/66
            #     dtype = hidden_states.dtype
            #     hidden_states = (hidden_states *
            #                      attention_mask[:, :, None]).to(dtype)
            scan_output, ssm_state = mamba_chunk_scan_combined(
                hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                time_step,
                self.A,
                B.view(batch_size, seq_len, self.n_groups, -1),
                C.view(batch_size, seq_len, self.n_groups, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=None,
                return_final_states=True,
                **dt_limit_kwargs,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_state.copy_(ssm_state)
            scan_output = scan_output.view(batch_size, seq_len, -1)
            # Multiply "gate" branch and apply extra normalization layer
            scan_output = self.norm(scan_output, gate)
            out = self.out_proj(scan_output)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        if attn_metadata.prefill_metadata is not None:
            offset = 0
            for i, prompt_len in enumerate(
                    attn_metadata.prefill_metadata.seq_lens):
                cache = MambaCacheParams(True,
                                         conv_state=conv_state[i].unsqueeze(0),
                                         ssm_state=ssm_state[i].unsqueeze(0))
                hidden_states[offset:offset + prompt_len].copy_(
                    self.mamba_forward(hidden_states[offset:offset +
                                                     prompt_len].unsqueeze(0),
                                       cache_params=cache)[0].squeeze(0))
                offset += prompt_len
        else:
            cache = MambaCacheParams(False,
                                     conv_state=conv_state,
                                     ssm_state=ssm_state)
            hidden_states = self.mamba_forward(hidden_states.unsqueeze(1),
                                               cache_params=cache)
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class MambaMLP(nn.Module):

    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        hidden_act = config.hidden_act
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: MambaConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.mixer = MambaMixer(config, layer_idx)

        self.feed_forward = MambaMLP(config, quant_config=quant_config)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states, attn_metadata, conv_state,
                                   ssm_state)
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class MambaModel(nn.Module):

    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            decoder_layers.append(
                MambaDecoderLayer(config,
                                  layer_idx=i,
                                  cache_config=cache_config,
                                  quant_config=quant_config))
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_f = RMSNorm(config.hidden_size,
                              eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            current_ssm_state = ssm_state[i]
            current_conv_state = conv_state[i]

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                conv_state=current_conv_state,
                ssm_state=current_ssm_state,
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states


class Mamba2ForCausalLM(nn.Module, HasInnerState):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embeddings": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: MambaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.backbone = MambaModel(config,
                                   cache_config=cache_config,
                                   quant_config=quant_config,
                                   lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs):
        if self.mamba_cache is None:
            max_batch_size = (_get_graph_batch_size(
                self.scheduler_config.max_num_seqs) if self.scheduler_config
                              else max(_BATCH_SIZES_TO_CAPTURE) + 2)
            self.mamba_cache = MambaCacheManager(
                self.lm_head.weight.dtype, self.config.num_hidden_layers,
                max_batch_size, *self._get_mamba_cache_shape())

        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            assert all(
                key in kwargs
                for key in ["request_ids_to_seq_ids", "finished_requests_ids"])

            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]
            self.mamba_cache.release_finished_requests(finished_requests_ids)

            batch_size = input_ids.shape[0]
            if attn_metadata.prefill_metadata:
                batch_size = len(request_ids_to_seq_ids)
            mamba_cache_tensors = self.mamba_cache.prepare_current_run_state(
                request_ids_to_seq_ids, batch_size, finished_requests_ids)

        else:
            # CUDA graph capturing runs
            mamba_cache_tensors = kwargs["seqlen_agnostic_capture_inputs"]

        hidden_states = self.backbone(input_ids, positions, kv_caches,
                                      attn_metadata, mamba_cache_tensors[0],
                                      mamba_cache_tensors[1])

        return hidden_states

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        conv_dim = (self.config.intermediate_size +
                    2 * self.config.n_groups * self.config.state_size)
        conv_state_shape = (
            conv_dim // world_size,
            self.config.conv_kernel,
        )
        temporal_state_shape = (
            self.config.num_heads // world_size,
            self.config.head_dim,
            self.config.state_size,
        )
        return conv_state_shape, temporal_state_shape

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "A_log" in name:
                name = name.replace("A_log", "A")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
