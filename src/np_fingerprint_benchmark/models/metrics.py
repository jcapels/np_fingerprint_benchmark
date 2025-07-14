import torch
import torch.nn.functional as F
from torchmetrics import Metric

class MaskedSymbolRecoveryAccuracy(Metric):
    def __init__(self, tokenizer, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer  # Needed to identify special tokens
        
        # Store counts as torch tensors for distributed training compatibility
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, labels, logits, mask_indices):
        """
        input_ids: Tensor of shape (batch_size, seq_len), input tokens with masking
        labels: Tensor of shape (batch_size, seq_len), true token labels
        logits: Tensor of shape (batch_size, seq_len, vocab_size), model outputs
        mask_indices: Boolean tensor of shape (batch_size, seq_len), where True indicates masked positions
        """
        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Select only masked token positions
        masked_labels = labels[mask_indices]
        predicted_tokens = predicted_ids[mask_indices]
        
        # Count correct predictions
        correct_preds = (masked_labels == predicted_tokens).sum()
        total_preds = mask_indices.sum()

        # Update metric state
        self.correct += correct_preds
        self.total += total_preds

    def compute(self):
        """ Compute final accuracy score """
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)
