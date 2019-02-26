from torch.autograd import grad


class CriticUpdater:
    def __init__(self, critic, critic_optimizer, eps, ones, gp_lambda=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.ones = ones
        self.gp_lambda = gp_lambda

    def __call__(self, real, fake):
        real = real.detach()
        fake = fake.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()
        self.loss_value = loss.item()


def mask_norm(diff, mask):
    """Mask normalization"""
    dim = 1, 2, 3
    # Assume mask.sum(1) is non-zero throughout
    return ((diff * mask).sum(dim) / mask.sum(dim)).mean()


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau
