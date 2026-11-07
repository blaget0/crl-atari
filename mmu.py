import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMemoryUnit(nn.Module):
    def __init__(self, in_features: int, layer_size: int, mask_size: int, task_emb_size: int, num_tasks: int, out_features: int):
        super(MaskedMemoryUnit, self).__init__()

        self.in_features = in_features
        self.layer_size = layer_size
        self.mask_size = mask_size
        self.task_emb_size = task_emb_size
        self.num_tasks = num_tasks

        # mask loss weights
        self.alpha = 0.5
        self.beta = 0.5

        self.fc1 = nn.Linear(in_features, layer_size)
        self.task_embeddings = nn.Embedding(num_tasks, task_emb_size)
        self.mask = nn.Linear(in_features + task_emb_size, layer_size)
        self.fc2 = nn.Linear(mask_size, out_features)
        self.act = nn.GELU()

        self.register_buffer('task_mask_prototypes', torch.zeros(num_tasks, layer_size))
        self.register_buffer('task_prototypes_initialized', torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer('task_count', torch.zeros(num_tasks, dtype=torch.long))

    def forward(self, x, task_id):
        task_ids = torch.full((x.size(0),), task_id, dtype=torch.long, device=x.device)
        task_emb = self.task_embeddings(task_ids)
        mask_input = torch.cat([x, task_emb], dim=1)
        scores = self.mask(mask_input)          # (batch, layer_size)
        features = self.act(self.fc1(x))         # (batch, layer_size)

        k = self.mask_size
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)   # topk_idx: (batch, k)

        selected_features = torch.gather(features, 1, topk_idx)
        selected_scores = torch.gather(scores, 1, topk_idx)

        gate = torch.sigmoid(selected_scores)
        gated_selected = selected_features * gate

        output = self.fc2(gated_selected)

        mask_loss = self.alpha * self.get_divergence_loss(scores, task_id) + self.beta * self.get_batch_consistency_loss(scores)

        return output, mask_loss, scores

    def update_mask_prototype(self, task_id, scores):
        current_proto = scores.mean(dim=0).detach()

        #if self.task_mask_prototypes is None:
        #    self.task_mask_prototypes = torch.zeros((self.num_tasks, self.layer_size), device=scores.device)

        # exponential moving average
        if not self.task_prototypes_initialized[task_id]:
            self.task_mask_prototypes[task_id] = current_proto
            self.task_prototypes_initialized[task_id] = True
        else:
            self.task_mask_prototypes[task_id] = 0.9 * self.task_mask_prototypes[task_id] + 0.1 * current_proto
        self.task_count[task_id] = 1

    def get_divergence_loss(self, current_scores, current_task_id, margin=0.3):
        if isinstance(current_task_id, torch.Tensor):
            current_task_id = current_task_id.item()

        if current_task_id == 0 or not self.task_prototypes_initialized[:current_task_id].any():
            return torch.tensor(0.0, device=current_scores.device)

        current_proto = current_scores.mean(dim=0)

        prev_protos = self.task_mask_prototypes[:current_task_id]
        if prev_protos.shape[0] == 0:
            return torch.tensor(0.0, device=current_scores.device)

        current_expanded = current_proto.unsqueeze(0).expand(prev_protos.shape[0], -1)
        similarities = F.cosine_similarity(current_expanded, prev_protos, dim=1)

        divergence_loss = torch.mean(F.relu(similarities - margin))
        return divergence_loss

    def get_batch_consistency_loss(self, current_scores):
        if current_scores.shape[0] <= 1:
            return torch.tensor(0.0, device=current_scores.device)

        return torch.var(current_scores, dim=0).mean()