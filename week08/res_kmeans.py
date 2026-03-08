import torch
from torch import nn

class ResKmeansPT(nn.Module):
    def __init__(self, n_layers, codebook_size, dim, n_iters=20):
        super().__init__()
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.n_iters = n_iters
        # Кодбуки для каждого слоя
        self.centroids = nn.ParameterList([
            nn.Parameter(torch.zeros((codebook_size, dim), requires_grad=False))
            for _ in range(n_layers)
        ])
    
    def calc_loss(self, x, out, epsilon=1e-4):
        loss = ((out - x) ** 2).mean()
        rel_loss = (torch.abs(x - out) / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)).mean()
        return {'loss': loss.item(), 'rel_loss': rel_loss.item()}
    
    def train_kmeans(self, inputs, verbose=True):
        """
        inputs: torch.Tensor [n_samples, dim]
        """
        x = inputs.clone()
        out = torch.zeros_like(x)
        for l in range(self.n_layers):
            # --- Инициализация центроидов случайными точками из x ---
            idx = torch.randperm(x.shape[0])[:self.codebook_size]
            centroids = x[idx].clone()
            
            for it in range(self.n_iters):
                # Вычисляем расстояния до центроидов
                dists = torch.cdist(x, centroids, p=2)  # [n_samples, codebook_size]
                codes = dists.argmin(dim=1)             # [n_samples]
                
                # Обновляем центроиды как среднее
                for k in range(self.codebook_size):
                    mask = codes == k
                    if mask.any():
                        centroids[k] = x[mask].mean(dim=0)
            
            # Сохраняем слой
            self.centroids[l] = nn.Parameter(centroids.clone(), requires_grad=False)
            
            # Вычисляем остаток
            dists = torch.cdist(x, centroids, p=2)
            codes = dists.argmin(dim=1)
            o = centroids[codes]
            out += o
            x = x - o
            
            if verbose:
                losses = self.calc_loss(inputs, out)
                print(f"Layer {l} finished, loss={losses}")
    
    def encode(self, x, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers
        else:
            assert n_layers <= self.n_layers
        
        codes_list = []
        for l in range(n_layers):
            dists = torch.cdist(x, self.centroids[l], p=2)
            codes = dists.argmin(dim=1)
            codes_list.append(codes)
            x = x - self.centroids[l][codes]
        
        return torch.stack(codes_list, dim=1)
    
    def decode(self, code):
        out = torch.zeros((code.shape[0], self.dim), dtype=torch.float32, device=code.device)
        n_layers = code.shape[1]
        for l in range(n_layers):
            out += self.centroids[l][code[:, l]]
        return out