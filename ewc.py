import torch
import torch.nn as nn



class EWC(nn.Module):
    def __init__(self, agent, lambda_ewc):
        super().__init__()
        object.__setattr__(self, "agent", agent)
        object.__setattr__(self, "model", agent.dqn)
        #self.agent = agent
        #self.model = agent.dqn
        self.lambda_ewc = lambda_ewc
        self.device = self.agent.device

    def estimate_fisher(self, batch_size, num_batches):
        fisher = {n.replace('.', '__'): torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.train()

        for _ in range(num_batches):
            samples = self.agent.memory.sample_batch(self.agent.beta)
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)

            self.model.zero_grad()
            elementwise_loss = self.agent._compute_dqn_loss(samples)
            loss = torch.mean(elementwise_loss * weights)

            loss.backward()

            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.grad is not None:
                    fisher[n] += p.grad ** 2

        for n in fisher:
            fisher[n] = fisher[n] / num_batches

        return fisher

    # online ewc
    def consolidate(self, fisher, gamma=0.9):
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            mean_name = f"mean_{n}"
            fisher_name = f"fisher_{n}"

            self.register_buffer(mean_name, p.data.clone().detach())

            new_fisher = fisher[n].clone().detach()
            if hasattr(self, fisher_name):
                old_fisher = getattr(self, fisher_name)
                new_fisher = gamma * old_fisher + new_fisher

            self.register_buffer(fisher_name, new_fisher)

    def ewc_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')

            # no ewc loss on MMU layers
            if "MMU" in n:
                continue

            mean_name = f"mean_{n}"
            fisher_name = f"fisher_{n}"
            if hasattr(self, mean_name):
                mean = getattr(self, mean_name)
                fisher = getattr(self, fisher_name)
                loss += (fisher * (p - mean) ** 2).sum()
        return (self.lambda_ewc / 2) * loss

    def clear_buffers(self):
        for name in list(self._buffers.keys()):
            delattr(self, name)


    def state_dict(self, *args, **kwargs):
        return {name: buf.detach().clone()
                for name, buf in self._buffers.items()}

    def load_state_dict(self, state_dict, strict=True):
        for name, buf in state_dict.items():
            self.register_buffer(name, buf.detach().clone())
        return super().load_state_dict(state_dict, strict=strict)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, strict=True):
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, strict=strict)