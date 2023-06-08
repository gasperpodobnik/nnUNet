import torch

class KD_cost(nn.Module):
    def __init__(self, n_class):
        self.n_class = n_class
        
    def forward(self, source_logits, source_gt, target_logits):
        target_gt = source_gt
        kd_loss = 0.0

        self.source_logits = source_logits
        self.target_logits = target_logits

        self.source_prob = []
        self.target_prob = []

        temperature = 2.0
        eps = 1e-6
        
        # source_gt: batch_size, 1, patch_size, patch_size, patch_size

        for i in range(self.n_class):

            self.s_mask = (source_gt==i).repeat(1, self.n_class, 1, 1, 1)
            self.s_logits_mask_out = self.source_logits * self.s_mask
            self.s_logits_avg = torch.sum(self.s_logits_mask_out, dim=[0, 2, 3, 4]) / (torch.sum(source_gt==i) + eps)
            self.s_soft_prob = torch.nn.functional.softmax(self.s_logits_avg / temperature, dim=-1)

            self.source_prob.append(self.s_soft_prob)

            self.t_mask = (target_gt==i).repeat(1, self.n_class, 1, 1, 1)
            self.t_logits_mask_out = self.target_logits * self.t_mask
            self.t_logits_avg = torch.sum(self.t_logits_mask_out, dim=[0, 2, 3, 4]) / (torch.sum(target_gt==i) + eps)
            self.t_soft_prob = torch.nn.functional.softmax(self.t_logits_avg / temperature, dim=-1)

            self.target_prob.append(self.t_soft_prob)

            ## KL divergence loss
            loss = (torch.sum(self.s_soft_prob * torch.log(self.s_soft_prob / self.t_soft_prob)) +
                    torch.sum(self.t_soft_prob * torch.log(self.t_soft_prob / self.s_soft_prob))) / 2.0
            
            

            ## L2 Norm
            # loss = torch.nn.functional.mse_loss(self.s_soft_prob, self.t_soft_prob) / self.n_class

            kd_loss += loss

        return kd_loss / self.n_class