import logging
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import deepspeed
import torch
from accelerate.utils import (
    broadcast_object_list,
    is_peft_model,
)
from accelerate.utils.memory import clear_device_cache
from peft import PeftConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer

import verifiers as vf
import wandb
from verifiers.rl.inference.client import VLLMClient
from verifiers.rl.trainer.config import RLConfig
from verifiers.rl.trainer.generator import Generator
from verifiers.rl.trainer.utils import (
    entropy_from_logits,
    finalize_stat_tracker,
    init_stat_tracker,
    pad,
    prepare_peft_model,
    selective_log_softmax,
    summarize_values,
    update_stat_tracker,
)
from verifiers.types import Messages
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls


class RLTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | str,
        env: vf.Environment,
        args: RLConfig,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        # model + tokenizer
        if isinstance(model, str):
            model_name = model
            model, processing_class = vf.get_model_and_tokenizer(model)
        else:
            model_name = model.config._name_or_path
        assert isinstance(model, PreTrainedModel)
        if args.use_lora and isinstance(args.lora_config, PeftConfig):
            model = prepare_peft_model(model, args.lora_config, args)
        model.warnings_issued["estimate_tokens"] = True  # suppress warning

        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            **kwargs,
        )
        assert isinstance(self.processing_class, PreTrainedTokenizerBase)
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id
        assert self.processing_class.pad_token_id is not None

        # batch args
        self.batch_size = args.batch_size
        self.max_steps = args.max_steps
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature

        # loss args
        self.mask_ratio_low = args.mask_ratio_low
        self.mask_ratio_high = args.mask_ratio_high

        # generator (main process only)
        if self.accelerator.is_main_process:
            host = args.vllm_server_host
            port = args.vllm_server_port
            nccl_port = getattr(args, "nccl_group_port", 51216)
            self.client = VLLMClient(
                host=host, port=port, connection_timeout=args.vllm_server_timeout,
                group_port=nccl_port,
            )
            if not getattr(args, "skip_weight_sync", False):
                self.client.init_communicator()
            else:
                self.logger.info("skip_weight_sync=True: skipping NCCL init, vLLM weights won't be updated.")
            vllm_base_url = f"http://{host}:{port}/v1"
            self.generator = Generator(
                env=env,
                client_base_url=vllm_base_url,
                client_api_key="EMPTY",
                client_limit=args.max_concurrent,
                client_timeout=args.generation_timeout,
                model_name=model_name,
                sampling_args=dict(args.sampling_args),
                rollouts_per_example=args.rollouts_per_example,
                batch_size=args.batch_size,
                micro_batch_size=args.micro_batch_size,
                num_processes=self.accelerator.num_processes,
                generation_timeout=args.generation_timeout,
                processing_class=self.processing_class,
                mask_env_responses=args.mask_env_responses,
                max_seq_len=self.max_seq_len,
                max_prompt_len=args.max_prompt_len or self.max_seq_len,
                mask_truncated_completions=args.mask_truncated_completions,
                zero_truncated_completions=args.zero_truncated_completions,
                max_concurrent=args.max_concurrent,
                use_terminal_contrastive=args.use_terminal_contrastive,
                terminal_sim_thresh_pos=args.terminal_sim_thresh_pos,
                terminal_sim_thresh_neg=args.terminal_sim_thresh_neg,
                terminal_sim_measure=args.terminal_sim_measure,
                use_variational_tc=getattr(args, "use_variational_tc", False),
                use_vector_tc=getattr(args, "use_vector_tc", False),
            )
            self.generator.start()
            self.generator.submit_batch(0)
        else:
            self.generator = None
            self.client = None

        # metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._textual_logs = {
            "prompt": deque(),
            "completion": deque(),
            "rewards": defaultdict(lambda: deque()),
        }

    def training_step(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.update_vllm()
        if self.generator:
            self.generator.submit_batch(self.state.global_step + 1)

        broadcast_list = [None]
        if self.generator:
            broadcast_list = [self.generator.get_batch(self.state.global_step)]
        broadcast_object_list(broadcast_list)
        assert broadcast_list[0] is not None
        batch = broadcast_list[0]

        model.train()
        total_loss = torch.zeros((), device=self.accelerator.device)
        local_microbatches = batch.microbatches[self.accelerator.process_index]

        if batch.global_item_count <= 0:
            return total_loss

        world_size = max(self.accelerator.num_processes, 1)
        # ddp/zero3 average gradients across ranks, so scale by per-rank items
        tokens_per_rank = torch.tensor(
            float(batch.global_item_count) / float(world_size),
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        inv_tokens_per_rank = tokens_per_rank.reciprocal()
        ir_tracker = init_stat_tracker(self.accelerator.device)
        entropy_tracker = init_stat_tracker(self.accelerator.device)
        mismatch_kl_tracker = init_stat_tracker(self.accelerator.device)
        barlow_tracker = init_stat_tracker(self.accelerator.device)
        vt_kl_tracker = init_stat_tracker(self.accelerator.device)
        device = self.accelerator.device
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        assert pad_token_id is not None

        # V2: sample λ ~ Dirichlet(α) once per step — same weighting for all microbatches
        use_vector_tc = getattr(self.args, "use_vector_tc", False)
        if use_vector_tc:
            K = getattr(self.args, "reward_vector_dim", 3)
            alpha = getattr(self.args, "dirichlet_alpha", 1.0)
            concentration = torch.full((K,), alpha, device=device)
            lambda_weights = torch.distributions.Dirichlet(concentration).sample()
        else:
            lambda_weights = None

        for microbatch in local_microbatches:
            input_ids = pad(
                [torch.tensor(x, device=device) for x in microbatch.input_ids],
                padding_value=pad_token_id,  # type: ignore :(
                padding_side="right",
            )
            loss_mask = pad(
                [torch.tensor(x, device=device) for x in microbatch.loss_mask],
                padding_side="right",
            )
            inference_logprobs = pad(
                [torch.tensor(x, device=device) for x in microbatch.sampling_logprobs],
                padding_value=0,
                padding_side="right",
            )
            advantages = pad(
                [torch.tensor(x, device=device) for x in microbatch.advantages],
                padding_value=0,
                padding_side="right",
            )
            attn_mask = input_ids.ne(pad_token_id).int()
            use_barlow = getattr(self.args, "use_barlow_diversity", False)
            if use_barlow:
                trainer_logprobs, entropies, hidden_states = self.get_logprobs(
                    model, input_ids, attn_mask, return_hidden=True
                )
            else:
                trainer_logprobs, entropies = self.get_logprobs(model, input_ids, attn_mask)
                hidden_states = None
            loss_mask = loss_mask[:, 1:]
            inference_logprobs = inference_logprobs[:, 1:]
            advantages = advantages[:, 1:]
            # V2: build reward_vectors tensor from microbatch if present
            rv_list = getattr(microbatch, "reward_vectors", [])
            if use_vector_tc and rv_list and rv_list[0]:
                reward_vectors_t = torch.tensor(rv_list, dtype=torch.float32, device=device)
            else:
                reward_vectors_t = None

            mb_inputs = {
                "loss_mask": loss_mask,
                "inference_logprobs": inference_logprobs,
                "trainer_logprobs": trainer_logprobs,
                "entropies": entropies,
                "advantages": advantages,
                "positive_pair_indices": microbatch.positive_pair_indices,
                "negative_pair_indices": microbatch.negative_pair_indices,
                "hidden_states": hidden_states,
                "reward_vectors": reward_vectors_t,
                "lambda_weights": lambda_weights,
            }
            with self.compute_loss_context_manager():
                loss, summaries = self.compute_loss(
                    model,
                    mb_inputs,
                    num_items_in_batch=torch.tensor(self.batch_size, device=device),
                    return_outputs=True,
                )
            self.accelerator.backward(loss * inv_tokens_per_rank)
            total_loss = total_loss + (loss.detach() * inv_tokens_per_rank)
            assert isinstance(summaries, dict)
            update_stat_tracker(ir_tracker, summaries["importance_sampling"])
            update_stat_tracker(entropy_tracker, summaries["entropy"])
            update_stat_tracker(mismatch_kl_tracker, summaries["mismatch_kl"])
            update_stat_tracker(barlow_tracker, summaries["barlow_loss"])
            update_stat_tracker(vt_kl_tracker, summaries["vt_kl_loss"])

        ir_mean = finalize_stat_tracker(ir_tracker, self.accelerator)
        entropy_mean = finalize_stat_tracker(entropy_tracker, self.accelerator)
        mismatch_kl_mean = finalize_stat_tracker(mismatch_kl_tracker, self.accelerator)
        barlow_mean = finalize_stat_tracker(barlow_tracker, self.accelerator)
        vt_kl_mean = finalize_stat_tracker(vt_kl_tracker, self.accelerator)
        assert ir_mean is not None
        assert entropy_mean is not None
        assert mismatch_kl_mean is not None

        extra_metrics: dict[str, float] = {
            "importance_ratio": ir_mean,
            "entropy": entropy_mean,
            "mismatch_kl": mismatch_kl_mean,
            "barlow_loss": barlow_mean or 0.0,
            "vt_kl_loss": vt_kl_mean or 0.0,
        }

        if self.accelerator.is_main_process:
            metrics_to_log = {**batch.metrics_dict, **extra_metrics}
            self.log_metrics(
                mode="train",
                batch_metrics=metrics_to_log,
            )
            self.log_rollouts(
                prompts=batch.prompts,
                completions=batch.completions,
                rewards_dict=batch.rewards_dict,
            )

        self.maybe_clear_cache()
        return total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        loss_mask = inputs["loss_mask"].bool()
        entropies = inputs["entropies"]
        trainer_logprobs = inputs["trainer_logprobs"]
        inference_logprobs = inputs["inference_logprobs"]
        advantages = inputs["advantages"]
        positive_pair_indices: list[tuple[int, int]] = inputs.get("positive_pair_indices", [])
        negative_pair_indices: list[tuple[int, int]] = inputs.get("negative_pair_indices", [])
        hidden_states = inputs.get("hidden_states", None)

        # V2: vector lambda sampling — override advantages with R_i = λ · reward_vectors[i]
        reward_vectors = inputs.get("reward_vectors", None)  # (N, K) or None
        lambda_weights = inputs.get("lambda_weights", None)  # (K,) or None
        v2_z_per_rollout = None
        if getattr(self.args, "use_vector_tc", False) and reward_vectors is not None and lambda_weights is not None:
            R_raw = reward_vectors @ lambda_weights  # (N,)
            mu = R_raw.mean()
            sigma = R_raw.std().clamp(min=0.05)
            v2_z_per_rollout = (R_raw - mu) / sigma  # (N,)
            advantages = v2_z_per_rollout.unsqueeze(1).expand(-1, advantages.shape[1]).contiguous()

        log_importance_ratio = trainer_logprobs - inference_logprobs
        importance_ratio = torch.exp(log_importance_ratio)
        is_masked_low = importance_ratio < self.mask_ratio_low
        is_masked_high = importance_ratio > self.mask_ratio_high
        is_masked = is_masked_low | is_masked_high
        keep_mask = ~is_masked & loss_mask
        grpo_loss = (-importance_ratio * advantages)[keep_mask].sum()

        # Advantage-based contrastive loss (existing)
        if self.args.use_contrastive_loss:
            if self.args.contrastive_mode == "dpo":
                contrastive_loss = self.compute_contrastive_loss_dpo(
                    trainer_logprobs=trainer_logprobs,
                    advantages=advantages,
                    loss_mask=loss_mask,
                )
            else:
                contrastive_loss = self.compute_contrastive_loss_infonce(
                    trainer_logprobs=trainer_logprobs,
                    advantages=advantages,
                    loss_mask=loss_mask,
                )
            total_loss = grpo_loss + self.args.contrastive_weight * contrastive_loss
        else:
            total_loss = grpo_loss
            contrastive_loss = torch.tensor(0.0, device=grpo_loss.device)

        # Terminal contrastive loss (verifier-free, uses stdout similarity pairs)
        tc_loss = torch.tensor(0.0, device=grpo_loss.device)
        if self.args.use_terminal_contrastive and (positive_pair_indices or negative_pair_indices):
            tc_loss = self.compute_terminal_contrastive_loss(
                trainer_logprobs=trainer_logprobs,
                loss_mask=loss_mask,
                positive_pair_indices=positive_pair_indices,
                negative_pair_indices=negative_pair_indices,
            )
            total_loss = total_loss + self.args.terminal_contrastive_weight * tc_loss

        # Barlow Twins diversity regularization
        barlow_loss = torch.tensor(0.0, device=grpo_loss.device)
        if getattr(self.args, "use_barlow_diversity", False) and hidden_states is not None:
            barlow_loss = self.compute_barlow_diversity_loss(hidden_states, loss_mask)
            total_loss = total_loss + self.args.barlow_weight * barlow_loss

        # Variational TC KL regularizer: KL[N(z, 1) || N(0, 1)] = z²/2
        vt_kl_loss = torch.tensor(0.0, device=grpo_loss.device)
        if getattr(self.args, "use_vector_tc", False) and v2_z_per_rollout is not None:
            # V2: z already computed from reward_vectors @ lambda_weights
            vt_kl_loss = v2_z_per_rollout.pow(2).mean() / 2
            total_loss = total_loss + getattr(self.args, "variational_beta", 0.01) * vt_kl_loss
        elif getattr(self.args, "use_variational_tc", False):
            # V1: z_i is the normalized sim score stored as the advantage for rollout i.
            mask_f = loss_mask.float()
            token_counts = mask_f.sum(dim=1).clamp(min=1)
            z_per_rollout = (advantages * mask_f).sum(dim=1) / token_counts  # (N,)
            vt_kl_loss = z_per_rollout.pow(2).mean() / 2
            total_loss = total_loss + getattr(self.args, "variational_beta", 0.01) * vt_kl_loss

        mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1

        with torch.no_grad():
            ir_summary = summarize_values(importance_ratio[loss_mask])
            entropy_summary = summarize_values(entropies[loss_mask])
            mismatch_kl_summary = summarize_values(mismatch_kl[loss_mask])

        summaries = {
            "importance_sampling": ir_summary,
            "entropy": entropy_summary,
            "mismatch_kl": mismatch_kl_summary,
            "contrastive_loss": {"mean": contrastive_loss.detach()},
            "tc_loss": {"mean": tc_loss.detach()},
            "barlow_loss": summarize_values(barlow_loss.detach().unsqueeze(0)),
            "vt_kl_loss": summarize_values(vt_kl_loss.detach().unsqueeze(0)),
        }
        return total_loss, summaries

    def compute_contrastive_loss_infonce(
        self,
        trainer_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE-style contrastive loss with implicit hard negative mining.
        
        For each prompt group:
        - Positives: Rollouts with advantage > 0 (better than group mean)
        - Negatives: Rollouts with advantage < 0 (worse than group mean)
        - Loss: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        
        Args:
            trainer_logprobs: Log probabilities from current policy (batch_size, seq_len)
            advantages: Advantages per token (batch_size, seq_len)
            loss_mask: Mask for valid tokens (batch_size, seq_len)
        
        Returns:
            Contrastive loss scalar
        """
        # Compute per-sequence log probabilities (mean across tokens)
        seq_logprobs = (trainer_logprobs * loss_mask).sum(dim=1)
        seq_lengths = loss_mask.sum(dim=1).clamp(min=1)
        seq_logprobs = seq_logprobs / seq_lengths
        
        # Compute per-sequence advantages (mean across tokens)
        seq_advantages = (advantages * loss_mask).sum(dim=1) / seq_lengths
        
        # Reshape into groups (num_prompts, rollouts_per_example)
        batch_size = seq_logprobs.shape[0]
        rollouts_per_example = getattr(self.args, "rollouts_per_example", 16)
        
        if batch_size % rollouts_per_example != 0:
            # Can't form complete groups, skip contrastive loss
            return torch.tensor(0.0, device=seq_logprobs.device)
        
        num_prompts = batch_size // rollouts_per_example
        grouped_logprobs = seq_logprobs.view(num_prompts, rollouts_per_example)
        grouped_advantages = seq_advantages.view(num_prompts, rollouts_per_example)
        
        # Compute contrastive loss for each group
        losses = []
        temperature = self.args.contrastive_temperature
        
        for group_logprobs, group_advs in zip(grouped_logprobs, grouped_advantages):
            # Identify positives and negatives based on advantages
            positive_mask = group_advs > 0
            negative_mask = group_advs < 0
            
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue  # Skip groups without clear winners/losers
            
            pos_logprobs = group_logprobs[positive_mask]
            neg_logprobs = group_logprobs[negative_mask]
            
            # InfoNCE loss for each positive against all negatives
            pos_scores = pos_logprobs / temperature
            neg_scores = neg_logprobs / temperature
            
            for pos_score in pos_scores:
                # log(exp(pos) / (exp(pos) + sum(exp(neg))))
                # = pos - log(exp(pos) + sum(exp(neg)))
                # = pos - logsumexp([pos, neg...])
                all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
                loss = -pos_score + torch.logsumexp(all_scores, dim=0)
                losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=seq_logprobs.device)
        
        return torch.stack(losses).mean()

    def compute_contrastive_loss_dpo(
        self,
        trainer_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO-style pairwise contrastive loss.
        
        For each group:
        - Find best (highest advantage) and worst (lowest advantage)
        - Loss: -log(sigmoid((logp_best - logp_worst) / temperature))
        
        This is simpler than InfoNCE and works well for direct comparisons.
        
        Args:
            trainer_logprobs: Log probabilities from current policy
            advantages: Advantages per token
            loss_mask: Mask for valid tokens
        
        Returns:
            DPO-style contrastive loss
        """
        import torch.nn.functional as F
        
        # Compute per-sequence scores
        seq_logprobs = (trainer_logprobs * loss_mask).sum(dim=1)
        seq_lengths = loss_mask.sum(dim=1).clamp(min=1)
        seq_logprobs = seq_logprobs / seq_lengths
        
        seq_advantages = (advantages * loss_mask).sum(dim=1) / seq_lengths
        
        # Reshape into groups
        batch_size = seq_logprobs.shape[0]
        rollouts_per_example = getattr(self.args, "rollouts_per_example", 16)
        
        if batch_size % rollouts_per_example != 0:
            return torch.tensor(0.0, device=seq_logprobs.device)
        
        num_prompts = batch_size // rollouts_per_example
        grouped_logprobs = seq_logprobs.view(num_prompts, rollouts_per_example)
        grouped_advantages = seq_advantages.view(num_prompts, rollouts_per_example)
        
        # Compute DPO loss for each group
        losses = []
        temperature = self.args.contrastive_temperature
        
        for group_logprobs, group_advs in zip(grouped_logprobs, grouped_advantages):
            # Get best and worst
            best_idx = group_advs.argmax()
            worst_idx = group_advs.argmin()
            
            if best_idx == worst_idx or group_advs[best_idx] <= group_advs[worst_idx]:
                continue  # Skip if no clear preference
            
            # DPO loss: -log(sigmoid((logp_best - logp_worst) / temp))
            logit_diff = (group_logprobs[best_idx] - group_logprobs[worst_idx]) / temperature
            loss = -F.logsigmoid(logit_diff)
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=seq_logprobs.device)

        return torch.stack(losses).mean()

    def compute_terminal_contrastive_loss(
        self,
        trainer_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        positive_pair_indices: list[tuple[int, int]],
        negative_pair_indices: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Verifier-free InfoNCE contrastive loss using terminal output similarity pairs.

        Positive pairs: rollouts with similar stdout/stderr (same outcome, different commands).
        Negative pairs: rollouts with dissimilar stdout (different outcomes).

        Both are computed by terminal_similarity.select_pairs() in the Generator, so no
        reward function or external verifier is needed here.
        """
        seq_logprobs = (trainer_logprobs * loss_mask).sum(dim=1)
        seq_lengths = loss_mask.sum(dim=1).clamp(min=1)
        seq_logprobs = seq_logprobs / seq_lengths

        temperature = self.args.terminal_contrastive_temperature
        negative_set = set(
            (min(i, j), max(i, j)) for i, j in negative_pair_indices
        )

        losses = []
        for pos_i, pos_j in positive_pair_indices:
            pos_score = (seq_logprobs[pos_i] + seq_logprobs[pos_j]) / (2.0 * temperature)
            neg_scores = [
                seq_logprobs[k] / temperature
                for k in range(seq_logprobs.shape[0])
                if k != pos_i and k != pos_j
                and (min(pos_i, k), max(pos_i, k)) in negative_set
            ]
            if not neg_scores:
                continue
            all_scores = torch.cat([pos_score.unsqueeze(0), torch.stack(neg_scores)])
            loss = -pos_score + torch.logsumexp(all_scores, dim=0)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=trainer_logprobs.device)

        return torch.stack(losses).mean()

    def compute_barlow_diversity_loss(
        self,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Barlow Twins diversity regularization on rollout hidden-state embeddings.

        Pools last-layer hidden states over response tokens (using loss_mask) to get
        one embedding per rollout, then computes:
            L = sum((C_ii - 1)^2) + lambda * sum_{i≠j}(C_ij^2)
        where C is the (D, D) cross-correlation matrix across the N rollouts in the
        microbatch.

        With micro_batch_size=rollouts_per_example=8, all N rollouts are from the
        same task — this directly regularizes within-task output diversity.
        """
        # hidden_states: (N, L, H); loss_mask: (N, L) bool
        N, L, H = hidden_states.shape
        if N < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        # Mean-pool over response tokens for each rollout
        mask_f = loss_mask.float().unsqueeze(-1)          # (N, L, 1)
        token_counts = mask_f.sum(dim=1).clamp(min=1)     # (N, 1)
        embeddings = (hidden_states * mask_f).sum(dim=1) / token_counts  # (N, H)

        # L2-normalize per rollout → cosine similarity space
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = embeddings / norms  # (N, H), unit vectors

        # (N, N) rollout cosine-similarity matrix
        C = z @ z.T  # values in [-1, 1]

        # Off-diagonal: penalize pairwise similarity (want diverse rollout embeddings)
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=hidden_states.device)
        off_diag = C[off_diag_mask].pow(2).mean()

        return off_diag

    def get_logprobs(
        self,
        model,
        input_ids,
        attention_mask,
        batch_size=None,
        return_hidden=False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch_size or input_ids.size(0)  # chunking for memory peak
        all_logprobs = []
        all_entropies = []
        all_hidden = [] if return_hidden else None
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits_to_keep = attention_mask_batch.size(1) + 1
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                output_hidden_states=return_hidden,
            )
            logits = outputs.logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            targets = input_ids_batch[:, 1:]
            logits = logits[:, -logits_to_keep:]
            logits = logits / self.temperature
            logprobs = selective_log_softmax(logits, targets)
            entropies = entropy_from_logits(logits)
            all_logprobs.append(logprobs)
            all_entropies.append(entropies)
            if return_hidden:
                # last hidden layer, shifted to align with logits (positions 1..L)
                hidden = outputs.hidden_states[-1][:, 1:, :]  # (B, L-1, H)
                all_hidden.append(hidden)
        logprobs = torch.cat(all_logprobs, dim=0)
        entropies = torch.cat(all_entropies, dim=0)
        if return_hidden:
            return logprobs, entropies, torch.cat(all_hidden, dim=0)
        return logprobs, entropies

    def update_vllm(self):
        assert self.model is not None
        is_generating = False
        if self.generator:
            is_generating = self.generator.is_generating
        is_generating_list = [is_generating]
        broadcast_object_list(is_generating_list, from_process=0)
        is_generating = is_generating_list[0]

        waits = 0
        while is_generating:
            time.sleep(0.5)
            waits += 1
            if waits % 10 == 0:
                self.logger.info("Waiting for generation to finish before syncing.")
            if self.generator:
                is_generating = self.generator.is_generating
            is_generating_list = [is_generating]
            broadcast_object_list(is_generating_list, from_process=0)
            is_generating = is_generating_list[0]

        if self.state.global_step > 0 and not getattr(self.args, "skip_weight_sync", False):  # skip first step
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            zero_stage_3 = (
                deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
            )
            if zero_stage_3:
                gather_if_zero3 = deepspeed.zero.GatheredParameters
            else:
                gather_if_zero3 = nullcontext
            self.accelerator.wait_for_everyone()
            self.logger.info("Starting weight sync to vLLM")

            if is_peft_model(self.model):
                # PEFT: gather + merge, then update each parameter
                with gather_if_zero3(list(self.model.parameters())):
                    self.model.merge_adapter()  # type: ignore :(
                    for name, param in self.model.named_parameters():
                        # recover original parameter names
                        name = name.removeprefix("base_model.model.").replace(
                            ".base_layer", ""
                        )
                        if self.model.prefix in name:  # type: ignore :(
                            continue  # discard some parameters
                        if "original_module" in name:  # from modules_to_save
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.client:
                            self.client.update_named_param(name, param.data)
                    self.model.unmerge_adapter()  # type: ignore :(
            else:
                # non-PEFT models: gather + update each parameter individually
                for name, param in self.model.named_parameters():  # type: ignore :(
                    with gather_if_zero3([param]):
                        if self.client:
                            self.client.update_named_param(name, param.data)

            # reset cache + wait for background tasks to complete
            if self.client:
                self.client.reset_prefix_cache()
                while self.client.get_num_background_tasks() > 0:
                    time.sleep(0.5)
                    self.logger.info("Resetting prefix cache.")

            # free NCCL temporary buffers before next backward pass
            clear_device_cache()

        self.accelerator.wait_for_everyone()

    def get_train_dataloader(self):
        class StepsDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {"labels": 0}

        return DataLoader(StepsDataset(self.max_steps))

    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # cleanup
            if self.generator:
                self.generator.stop()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process:
            print_prompt_completions_sample(
                self._textual_logs["prompt"],
                self._textual_logs["completion"],
                self._textual_logs["rewards"]["reward"],
                self.state.global_step,
            )

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                def role_content_only(messages):
                    if isinstance(messages, str):
                        return messages
                    return [
                        {
                            "role": m.get("role", ""),
                            "content": m.get("content", ""),
                        }
                        for m in messages
                    ]

                prompts_clean = [
                    role_content_only(sanitize_tool_calls(messages_to_printable(p)))
                    for p in self._textual_logs["prompt"]
                ]
                completions_clean = [
                    role_content_only(sanitize_tool_calls(messages_to_printable(c)))
                    for c in self._textual_logs["completion"]
                ]
                table = {
                    "step": [str(self.state.global_step)]
                    * len(self._textual_logs["prompt"]),
                    "prompt": prompts_clean,
                    "completion": completions_clean,
                    **{k: list(v) for k, v in self._textual_logs["rewards"].items()},
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

            # clear after logging
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

    def log_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        rewards_dict: Dict[str, Any],
    ) -> None:
        self._textual_logs["prompt"].extend(prompts)
        self._textual_logs["completion"].extend(completions)
        for reward_key in rewards_dict:
            reward_values = rewards_dict[reward_key]
            self._textual_logs["rewards"][reward_key].extend(reward_values)

    def log_metrics(
        self,
        mode: str,
        batch_metrics: Dict[str, float],
    ) -> None:
        for key, value in batch_metrics.items():
            self._metrics[mode][key].append(value)

    def maybe_clear_cache(self):
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            clear_device_cache()
