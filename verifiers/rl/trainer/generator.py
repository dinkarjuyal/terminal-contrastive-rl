import asyncio
import logging
import queue
import threading
import time
from typing import Any

import httpx
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from verifiers import Environment
from verifiers.rl.trainer.terminal_similarity import compute_reward_vector, mean_sim_per_rollout, select_pairs, trajectory_diversity


class Microbatch(BaseModel):
    """Microbatch for batch generation"""

    input_ids: list[list[int]]
    loss_mask: list[list[int]]
    sampling_logprobs: list[list[float]]
    advantages: list[list[float]]
    items: int
    # terminal contrastive pairs (local indices within this microbatch)
    positive_pair_indices: list[tuple[int, int]] = Field(default_factory=list)
    negative_pair_indices: list[tuple[int, int]] = Field(default_factory=list)
    # vector TC reward dimensions (N, K), empty if not using vector TC
    reward_vectors: list[list[float]] = Field(default_factory=list)


class Batch(BaseModel):
    """Result from batch generation"""

    batch_id: int
    microbatches: list[list[Microbatch]]
    items_per_process: list[int]
    global_item_count: int
    # logging
    generation_time: float = 0.0
    prompts: list[Any] = Field(default_factory=list)
    completions: list[Any] = Field(default_factory=list)
    metrics_dict: dict[str, float] = Field(default_factory=dict)
    rewards_dict: dict[str, list[float]] = Field(default_factory=dict)
    # terminal contrastive stats (for logging)
    terminal_diversity: float = 0.0
    terminal_pair_rate: float = 0.0


class Generator:
    """
    Manages asynchronous batch generation in parallel with RL training.
    """

    def __init__(
        self,
        env: Environment,
        client_base_url: str,
        client_api_key: str,
        client_limit: int,
        client_timeout: float,
        model_name: str,
        sampling_args: dict[str, Any],
        rollouts_per_example: int,
        batch_size: int,
        micro_batch_size: int,
        num_processes: int,
        generation_timeout: float,
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool,
        max_seq_len: int,
        max_prompt_len: int,
        mask_truncated_completions: bool,
        zero_truncated_completions: bool,
        max_concurrent: int,
        use_terminal_contrastive: bool = False,
        terminal_sim_thresh_pos: float = 0.70,
        terminal_sim_thresh_neg: float = 0.20,
        terminal_sim_measure: str = "strict",
        use_variational_tc: bool = False,
        use_vector_tc: bool = False,
    ):
        self.env = env
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.client_limit = client_limit
        self.client_timeout = client_timeout
        self.client = None  # created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.rollouts_per_example = rollouts_per_example
        self.prompts_per_batch = batch_size // rollouts_per_example
        self.micro_batch_size = micro_batch_size
        self.num_processes = num_processes
        self.generation_timeout = generation_timeout
        self.processing_class = processing_class
        self.mask_env_responses = mask_env_responses
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.mask_truncated_completions = mask_truncated_completions
        self.zero_truncated_completions = zero_truncated_completions
        self.max_concurrent = max_concurrent
        self.use_terminal_contrastive = use_terminal_contrastive
        self.terminal_sim_thresh_pos = terminal_sim_thresh_pos
        self.terminal_sim_thresh_neg = terminal_sim_thresh_neg
        self.terminal_sim_measure = terminal_sim_measure
        self.use_variational_tc = use_variational_tc
        self.use_vector_tc = use_vector_tc

        # queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_generating = False
        self.completed_batches = {}

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.is_generating = False
        self.worker_loop = None

        max_length = self.max_prompt_len
        assert env.dataset is not None

        def filter_by_prompt_length(example, processing_class):
            prompt = example["prompt"]
            if isinstance(prompt, list):
                prompt_text = processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            enc = processing_class.encode(prompt_text); prompt_ids = enc.input_ids if hasattr(enc, "input_ids") else enc; prompt_ids = list(prompt_ids) if not isinstance(prompt_ids, list) else prompt_ids
            return len(prompt_ids) <= max_length

        env.dataset = env.dataset.filter(
            filter_by_prompt_length,
            fn_kwargs={"processing_class": processing_class},
        )

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """Get dataset slice for a given batch id"""
        num_rows = self.prompts_per_batch
        dataset = self.env.get_dataset()
        total_rows = len(dataset)
        if total_rows == 0:
            raise ValueError("Environment dataset is empty")
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]
        return dataset.select(indices)

    def start(self):
        """Start the async generation worker thread"""
        self.worker_thread = threading.Thread(
            target=self.generation_worker, daemon=True, name="BatchGenerator"
        )
        self.worker_thread.start()

    def stop(self):
        """Stop the async generation worker thread"""
        self.stop_event.set()
        self.request_queue.put(None)  # poison pill
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def submit_batch(self, batch_id: int):
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: batch doesn't complete within timeout
            RuntimeError: generation failed
        """
        timeout = self.generation_timeout
        start_time = time.time()
        while True:
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)
            try:
                result = self.result_queue.get(timeout=0.1)
                self.completed_batches[result.batch_id] = result
                if result.batch_id == batch_id:
                    return self.completed_batches.pop(batch_id)
            except queue.Empty:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")

    def generation_worker(self):
        """Worker thread that processes generation requests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop
        self.client = AsyncOpenAI(
            base_url=self.client_base_url,
            api_key=self.client_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.client_limit),
                timeout=self.client_timeout,
            ),
        )
        try:
            while not self.stop_event.is_set():
                try:
                    batch_id = self.request_queue.get(timeout=0.1)
                    if batch_id is None:  # poison pill
                        break
                    result = loop.run_until_complete(self.generate_batch(batch_id))
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    async def generate_batch(self, batch_id: int) -> Batch:
        """
        Generate a single batch asynchronously.
        """
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()
        try:
            batch_ds = self.get_dataset_slice(batch_id)
            repeated_ds = batch_ds.repeat(self.rollouts_per_example)
            env_results = await self.env.a_generate(
                repeated_ds,
                client=self.client,
                model=self.model_name,
                sampling_args=self.sampling_args,
                score_rollouts=True,
                max_concurrent=self.max_concurrent,
            )
        finally:
            self.is_generating = False
        wall_clock_s = time.time() - start_time

        prompts_in_batch = len(batch_ds)
        N = prompts_in_batch
        G = self.rollouts_per_example

        # Reorder from strided [p0r0, p1r0, ..., p0r1, p1r1, ...] to
        # interleaved [p0r0, p0r1, ..., p1r0, p1r1, ...] so that groups of G
        # consecutive sequences are all rollouts of the same prompt.
        # strided index for prompt p, rollout k: p + k * N
        # interleaved index for prompt p, rollout k: p * G + k
        # perm[interleaved_idx] = strided_idx
        total = N * G
        perm = [((i // G) + (i % G) * N) for i in range(total)]

        # Reorder env_results and processed_results lists
        reordered_prompts = [env_results.prompt[perm[i]] for i in range(total)]
        reordered_completions = [env_results.completion[perm[i]] for i in range(total)]
        reordered_states = [env_results.state[perm[i]] for i in range(total)]
        reordered_rewards = [env_results.reward[perm[i]] for i in range(total)]

        processed_results = self.env.process_env_results_vllm(
            prompts=reordered_prompts,
            completions=reordered_completions,
            states=reordered_states,
            rewards=reordered_rewards,
            processing_class=self.processing_class,
            max_seq_len=self.max_seq_len,
            mask_env_responses=self.mask_env_responses,
            mask_truncated_completions=self.mask_truncated_completions,
            zero_truncated_completions=self.zero_truncated_completions,
        )

        rewards_dict = {"reward": processed_results.rewards}
        for k in env_results.metrics:
            rewards_dict[k] = env_results.metrics[k]

        rewards: list[float] = processed_results.rewards
        advantages: list[float] = [0.0] * len(rewards)
        # Advantages in interleaved space: group [p*G, (p+1)*G) is all rollouts of prompt p
        for p in range(N):
            group_indices = list(range(p * G, min((p + 1) * G, len(rewards))))
            if not group_indices:
                continue
            group = [rewards[i] for i in group_indices]
            gmean = sum(group) / float(len(group))
            for idx, r in zip(group_indices, group):
                advantages[idx] = r - gmean

        # --- Terminal contrastive pair selection ---
        # For each prompt group, extract stdout from state and compute pairs.
        # Uses strided env_results.state (before reorder) to get rollouts per prompt.
        global_pos_pairs: list[tuple[int, int]] = []
        global_neg_pairs: list[tuple[int, int]] = []
        global_reward_vectors: list[list[float]] = [[]] * (N * G)
        diversity_scores: list[float] = []
        total_possible_pairs = 0
        use_tc = getattr(self, "use_terminal_contrastive", False)

        if use_tc:
            for p in range(N):
                # strided indices for this prompt (before reorder)
                strided_indices = [p + k * N for k in range(G) if (p + k * N) < len(env_results.state)]
                first_state = env_results.state[strided_indices[0]] if strided_indices else {}
                if "final_stdout" not in first_state:
                    continue  # env doesn't provide terminal outputs for this rollout

                stdouts = [env_results.state[i].get("final_stdout", "") for i in strided_indices]
                exit_codes = [env_results.state[i].get("exit_code", 0) for i in strided_indices]
                measure = getattr(self, "terminal_sim_measure", "strict")

                if getattr(self, "use_vector_tc", False):
                    # V2: store reward vectors; λ sampled fresh in trainer each step
                    rvecs = compute_reward_vector(stdouts, exit_codes=exit_codes)
                    for k, rv in enumerate(rvecs):
                        global_reward_vectors[p * G + k] = rv
                    # Use strict dim as proxy diversity metric
                    strict_scores = [rv[0] for rv in rvecs]
                    mu = sum(strict_scores) / len(strict_scores)
                    var = sum((s - mu) ** 2 for s in strict_scores) / len(strict_scores)
                    diversity_scores.append(max(var ** 0.5, 0.0))
                    total_possible_pairs += G * (G - 1) // 2
                elif getattr(self, "use_variational_tc", False):
                    # V1: continuous reward — replace GRPO advantages with normalized z_i
                    sim_scores = mean_sim_per_rollout(stdouts, exit_codes=exit_codes, measure=measure)
                    mu = sum(sim_scores) / len(sim_scores)
                    var = sum((s - mu) ** 2 for s in sim_scores) / len(sim_scores)
                    sigma = max(var ** 0.5, 0.05)
                    z_scores = [(s - mu) / sigma for s in sim_scores]
                    for k, z in enumerate(z_scores):
                        advantages[p * G + k] = z
                    diversity_scores.append(sigma)
                    total_possible_pairs += G * (G - 1) // 2
                else:
                    pos, neg = select_pairs(
                        stdouts,
                        exit_codes=exit_codes,
                        thresh_pos=getattr(self, "terminal_sim_thresh_pos", 0.70),
                        thresh_neg=getattr(self, "terminal_sim_thresh_neg", 0.20),
                        measure=measure,
                    )
                    # Map local (within-group) indices → global interleaved indices
                    global_pos_pairs += [(p * G + i, p * G + j) for i, j in pos]
                    global_neg_pairs += [(p * G + i, p * G + j) for i, j in neg]
                    diversity_scores.append(trajectory_diversity(stdouts, measure=measure))
                    total_possible_pairs += G * (G - 1) // 2

        metrics_dict = {}
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        # Log terminal contrastive metrics
        tc_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
        tc_pair_rate = len(global_pos_pairs) / max(1, total_possible_pairs)
        if use_tc:
            metrics_dict["tc/diversity"] = tc_diversity
            metrics_dict["tc/positive_pair_rate"] = tc_pair_rate
            metrics_dict["tc/pos_pairs"] = float(len(global_pos_pairs))
            metrics_dict["tc/neg_pairs"] = float(len(global_neg_pairs))

        for reward_name, values in env_results.metrics.items():
            if len(values) == 0:
                continue
            reward_values = np.asarray(values, dtype=np.float32)
            metrics_dict[f"reward/{reward_name}"] = float(reward_values.mean())

        completion_lengths = [len(ids) for ids in processed_results.completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

            completion_mask_lengths = np.asarray(
                [sum(mask) for mask in processed_results.completion_mask],
                dtype=np.float32,
            )
            valid_tokens = completion_mask_lengths.sum()
            total_tokens = completion_lengths_arr.sum()
            if total_tokens > 0:
                masked_fraction = 1.0 - (valid_tokens / total_tokens)
                metrics_dict["tokens/masked_fraction"] = float(masked_fraction)

        generation_ms: list[float] = []
        scoring_ms: list[float] = []
        total_ms: list[float] = []
        for state in env_results.state:
            timing = state.get("timing", {})
            if "generation_ms" in timing:
                generation_ms.append(float(timing["generation_ms"]))
            if "scoring_ms" in timing:
                scoring_ms.append(float(timing["scoring_ms"]))
            if "total_ms" in timing:
                total_ms.append(float(timing["total_ms"]))

        if generation_ms:
            metrics_dict["timing/generation_ms"] = float(np.mean(generation_ms))
        if scoring_ms:
            metrics_dict["timing/scoring_ms"] = float(np.mean(scoring_ms))
        if total_ms:
            metrics_dict["timing/total_ms"] = float(np.mean(total_ms))

        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)

        # build per-process microbatches
        total_seqs = len(processed_results.rewards)
        per_proc = total_seqs // self.num_processes
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []
        for proc in range(self.num_processes):
            ps = proc * per_proc
            pe = ps + per_proc
            proc_mbs: list[Microbatch] = []
            proc_item_total = 0
            for s in range(ps, pe, self.micro_batch_size):
                e = min(s + self.micro_batch_size, pe)
                ids_chunk = [
                    processed_results.prompt_ids[i]
                    + processed_results.completion_ids[i]
                    for i in range(s, e)
                ]
                mask_chunk = [
                    processed_results.prompt_mask[i]
                    + processed_results.completion_mask[i]
                    for i in range(s, e)
                ]
                slogp_chunk = [
                    [0.0] * len(processed_results.prompt_mask[i])
                    + processed_results.completion_logprobs[i]
                    for i in range(s, e)
                ]
                lengths = [len(mask) for mask in mask_chunk]
                adv_chunk = [
                    [advantages[i]] * lengths[idx]
                    for idx, i in enumerate(list(range(s, e)))
                ]
                mb_items = sum(sum(mask) for mask in mask_chunk)
                # Distribute contrastive pairs that fall within this microbatch's slice
                mb_pos = [(i - s, j - s) for i, j in global_pos_pairs if s <= i < e and s <= j < e]
                mb_neg = [(i - s, j - s) for i, j in global_neg_pairs if s <= i < e and s <= j < e]
                mb_rvecs = [global_reward_vectors[i] for i in range(s, e)]
                microbatch = Microbatch(
                    input_ids=ids_chunk,
                    loss_mask=mask_chunk,
                    sampling_logprobs=slogp_chunk,
                    advantages=adv_chunk,
                    items=mb_items,
                    positive_pair_indices=mb_pos,
                    negative_pair_indices=mb_neg,
                    reward_vectors=mb_rvecs,
                )
                proc_item_total += mb_items
                proc_mbs.append(microbatch)
            microbatches.append(proc_mbs)
            items_per_process.append(proc_item_total)

        global_item_count = sum(items_per_process)

        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=global_item_count,
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=reordered_completions,
            prompts=reordered_prompts,
            metrics_dict=metrics_dict,
            terminal_diversity=tc_diversity,
            terminal_pair_rate=tc_pair_rate,
        )
