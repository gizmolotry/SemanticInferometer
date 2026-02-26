import torch
cache = torch.load('outputs/nli_cache_86c75286.pt')
print('Feature shape:', cache.shape)
print('Mean feature std:', cache.std(dim=0).mean().item())
print('Feature variance:', cache.var(dim=0).mean().item())
print('Min/max feature values:', cache.min().item(), cache.max().item())