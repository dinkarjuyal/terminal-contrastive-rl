#!/usr/bin/env python3
"""
Test script for contrastive loss implementation.

This tests the contrastive loss computation without requiring GPUs or actual training.
Just verifies the math and logic work correctly.

Usage:
    python test_contrastive_loss.py
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
import sys


# Mock RLConfig with contrastive parameters
@dataclass
class MockRLConfig:
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    contrastive_mode: str = "infonce"
    rollouts_per_example: int = 8


class ContrastiveLossTest:
    """Test contrastive loss implementation."""
    
    def __init__(self):
        self.args = MockRLConfig()
    
    def compute_contrastive_loss_infonce(
        self,
        trainer_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss (copied from trainer.py)."""
        seq_logprobs = (trainer_logprobs * loss_mask).sum(dim=1)
        seq_lengths = loss_mask.sum(dim=1).clamp(min=1)
        seq_logprobs = seq_logprobs / seq_lengths
        
        seq_advantages = (advantages * loss_mask).sum(dim=1) / seq_lengths
        
        batch_size = seq_logprobs.shape[0]
        rollouts_per_example = self.args.rollouts_per_example
        
        if batch_size % rollouts_per_example != 0:
            return torch.tensor(0.0, device=seq_logprobs.device)
        
        num_prompts = batch_size // rollouts_per_example
        grouped_logprobs = seq_logprobs.view(num_prompts, rollouts_per_example)
        grouped_advantages = seq_advantages.view(num_prompts, rollouts_per_example)
        
        losses = []
        temperature = self.args.contrastive_temperature
        
        for group_logprobs, group_advs in zip(grouped_logprobs, grouped_advantages):
            positive_mask = group_advs > 0
            negative_mask = group_advs < 0
            
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue
            
            pos_logprobs = group_logprobs[positive_mask]
            neg_logprobs = group_logprobs[negative_mask]
            
            pos_scores = pos_logprobs / temperature
            neg_scores = neg_logprobs / temperature
            
            for pos_score in pos_scores:
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
        """DPO contrastive loss (copied from trainer.py)."""
        import torch.nn.functional as F
        
        seq_logprobs = (trainer_logprobs * loss_mask).sum(dim=1)
        seq_lengths = loss_mask.sum(dim=1).clamp(min=1)
        seq_logprobs = seq_logprobs / seq_lengths
        
        seq_advantages = (advantages * loss_mask).sum(dim=1) / seq_lengths
        
        batch_size = seq_logprobs.shape[0]
        rollouts_per_example = self.args.rollouts_per_example
        
        if batch_size % rollouts_per_example != 0:
            return torch.tensor(0.0, device=seq_logprobs.device)
        
        num_prompts = batch_size // rollouts_per_example
        grouped_logprobs = seq_logprobs.view(num_prompts, rollouts_per_example)
        grouped_advantages = seq_advantages.view(num_prompts, rollouts_per_example)
        
        losses = []
        temperature = self.args.contrastive_temperature
        
        for group_logprobs, group_advs in zip(grouped_logprobs, grouped_advantages):
            best_idx = group_advs.argmax()
            worst_idx = group_advs.argmin()
            
            if best_idx == worst_idx or group_advs[best_idx] <= group_advs[worst_idx]:
                continue
            
            logit_diff = (group_logprobs[best_idx] - group_logprobs[worst_idx]) / temperature
            loss = -F.logsigmoid(logit_diff)
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=seq_logprobs.device)
        
        return torch.stack(losses).mean()


def create_mock_data():
    """Create mock data simulating a GRPO batch."""
    # 2 prompts, 8 rollouts each = 16 total rollouts
    batch_size = 16
    seq_len = 10
    
    # Mock logprobs (negative values, as they should be)
    trainer_logprobs = torch.randn(batch_size, seq_len) - 2.0
    
    # Mock advantages (some positive, some negative)
    # Group 1: 4 positive, 4 negative
    # Group 2: 3 positive, 5 negative
    advantages = torch.tensor([
        # Group 1 (prompt 1)
        [0.5] * seq_len,   # positive
        [0.3] * seq_len,   # positive
        [0.2] * seq_len,   # positive
        [0.1] * seq_len,   # positive
        [-0.1] * seq_len,  # negative
        [-0.2] * seq_len,  # negative
        [-0.3] * seq_len,  # negative
        [-0.5] * seq_len,  # negative
        # Group 2 (prompt 2)
        [0.4] * seq_len,   # positive
        [0.2] * seq_len,   # positive
        [0.1] * seq_len,   # positive
        [-0.1] * seq_len,  # negative
        [-0.2] * seq_len,  # negative
        [-0.3] * seq_len,  # negative
        [-0.4] * seq_len,  # negative
        [-0.6] * seq_len,  # negative
    ])
    
    # Loss mask (all valid)
    loss_mask = torch.ones(batch_size, seq_len)
    
    return trainer_logprobs, advantages, loss_mask


def test_infonce():
    """Test InfoNCE contrastive loss."""
    print("Testing InfoNCE contrastive loss...")
    
    tester = ContrastiveLossTest()
    tester.args.contrastive_mode = "infonce"
    
    logprobs, advantages, loss_mask = create_mock_data()
    
    loss = tester.compute_contrastive_loss_infonce(logprobs, advantages, loss_mask)
    
    print(f"  Loss value: {loss.item():.4f}")
    
    # Sanity checks
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 100, "Loss seems too large"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"
    
    print("  ✓ InfoNCE test passed!")
    return loss.item()


def test_dpo():
    """Test DPO contrastive loss."""
    print("\nTesting DPO contrastive loss...")
    
    tester = ContrastiveLossTest()
    tester.args.contrastive_mode = "dpo"
    
    logprobs, advantages, loss_mask = create_mock_data()
    
    loss = tester.compute_contrastive_loss_dpo(logprobs, advantages, loss_mask)
    
    print(f"  Loss value: {loss.item():.4f}")
    
    # Sanity checks
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 100, "Loss seems too large"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"
    
    print("  ✓ DPO test passed!")
    return loss.item()


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\nTesting gradient flow...")
    
    tester = ContrastiveLossTest()
    
    # Create mock data with requires_grad
    logprobs = torch.randn(16, 10, requires_grad=True) - 2.0
    advantages, loss_mask = create_mock_data()[1:]
    
    # Compute loss and backward
    loss = tester.compute_contrastive_loss_infonce(logprobs, advantages, loss_mask)
    loss.backward()
    
    # Check gradients exist and are reasonable
    assert logprobs.grad is not None, "No gradient computed"
    assert not torch.isnan(logprobs.grad).any(), "NaN in gradients"
    assert not torch.isinf(logprobs.grad).any(), "Inf in gradients"
    
    grad_norm = logprobs.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Zero gradients"
    assert grad_norm < 1000, "Gradient norm too large"
    
    print("  ✓ Gradient flow test passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    tester = ContrastiveLossTest()
    
    # Case 1: All positive advantages
    logprobs = torch.randn(16, 10) - 2.0
    advantages = torch.ones(16, 10) * 0.5  # All positive
    loss_mask = torch.ones(16, 10)
    
    loss = tester.compute_contrastive_loss_infonce(logprobs, advantages, loss_mask)
    assert loss.item() == 0.0, "Should be zero when no negatives"
    print("  ✓ All positive case: OK")
    
    # Case 2: All negative advantages
    advantages = torch.ones(16, 10) * -0.5  # All negative
    loss = tester.compute_contrastive_loss_infonce(logprobs, advantages, loss_mask)
    assert loss.item() == 0.0, "Should be zero when no positives"
    print("  ✓ All negative case: OK")
    
    # Case 3: Wrong batch size (not divisible)
    logprobs = torch.randn(15, 10) - 2.0  # 15 not divisible by 8
    advantages = torch.randn(15, 10)
    loss_mask = torch.ones(15, 10)
    loss = tester.compute_contrastive_loss_infonce(logprobs, advantages, loss_mask)
    assert loss.item() == 0.0, "Should be zero when batch size wrong"
    print("  ✓ Wrong batch size: OK")
    
    print("  ✓ Edge cases passed!")


def test_temperature_effect():
    """Test that temperature affects loss."""
    print("\nTesting temperature effect...")
    
    tester = ContrastiveLossTest()
    logprobs, advantages, loss_mask = create_mock_data()
    
    # Test different temperatures
    temps = [0.01, 0.1, 0.5, 1.0]
    losses = []
    
    for temp in temps:
        tester.args.contrastive_temperature = temp
        loss = tester.compute_contrastive_loss_infonce(logprobs, advantages, loss_mask)
        losses.append(loss.item())
        print(f"  Temperature {temp}: loss={loss.item():.4f}")
    
    # Lower temperature should give higher loss (sharper distinctions)
    assert losses[0] > losses[-1], "Lower temp should give higher loss"
    print("  ✓ Temperature effect verified!")


def main():
    """Run all tests."""
    print("="*60)
    print("CONTRASTIVE LOSS IMPLEMENTATION TEST")
    print("="*60)
    
    try:
        # Basic functionality tests
        infonce_loss = test_infonce()
        dpo_loss = test_dpo()
        
        # Gradient flow test
        test_gradient_flow()
        
        # Edge cases
        test_edge_cases()
        
        # Temperature effect
        test_temperature_effect()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nSummary:")
        print(f"  InfoNCE loss: {infonce_loss:.4f}")
        print(f"  DPO loss: {dpo_loss:.4f}")
        print("\nContrastive loss implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Try on actual training: configs/rl/contrastive_example.toml")
        print("  2. Monitor 'contrastive_loss/mean' in W&B")
        print("  3. Compare with baseline (use_contrastive_loss=false)")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())




