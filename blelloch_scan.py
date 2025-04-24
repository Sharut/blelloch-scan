import math
from typing import Callable, Optional
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn


class BlellochScan(nn.Module):
    """
    Parallel inclusive/exclusive scan (prefix operation) using the Blelloch algorithm.

    Args:
        combine_fn: Callable merging two blocks of shape (B, D, N) -> (B, D, N).
        identity: Optional identity tensor of shape (D, N). Defaults to zero.
        inclusive: If True, returns inclusive scan; otherwise, exclusive scan.
    """

    def __init__(
        self,
        combine_fn: Callable[[Tensor, Tensor], Tensor],
        identity: Optional[Tensor] = None,
        inclusive: bool = True,
    ) -> None:
        super().__init__()
        self.combine_fn = combine_fn
        self.identity = identity
        self.inclusive = inclusive

    def _make_identity(
        self, B: int, D: int, N: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        if self.identity is not None:
            return self.identity.to(device).expand(D, N)
        return torch.zeros(D, N, dtype=dtype, device=device)
    
    def forward(self, X_in):
        B, L, D, N = X_in.shape                                         # X_in: (B, L, D, N)
        P = 1 << (L - 1).bit_length()                                   # next power of two >= L

        X = X_in.transpose(1, 2).contiguous()                           # reshape to (B, D, L, N)
        id_val = self._make_identity(B, D, N, X.dtype, X.device)

        # pad to length P
        if P != L:
            pad = id_val.unsqueeze(2).expand(B, D, P - L, N)
            X = torch.cat([X, pad], dim=2)


        X_orig = X.clone()
        levels = int(math.log2(P))

        # --- upsweep (reduction) ---
        X = self._upsweep(X, levels)

        # --- downsweep (distribution) ---
        X = self._downsweep(X, levels, id_val)

        # remove padding and transpose back to (B, L, D, N)
        if self.inclusive:
            X = self.combine_fn(X, X_orig)
            return X[:, :, :L, :].transpose(2, 1)   
        return X[:, :, :L, :].transpose(2, 1)   

    def _upsweep(self, X: Tensor, levels: int) -> Tensor:
        B, D, P, N = *X.shape[:3], X.shape[3]
        for lvl in range(levels):
            step = 2 ** lvl
            idx_l = torch.arange(step - 1, P, 2 * step, device=X.device)
            idx_r = idx_l + step
            # gather blocks
            left = X[:, :, idx_l, :]
            right = X[:, :, idx_r, :]
            # merge
            Bk = left.shape[0] * left.shape[2]
            left_flat = left.permute(0, 2, 1, 3).reshape(Bk, D, N)
            right_flat = right.permute(0, 2, 1, 3).reshape(Bk, D, N)
            merged_flat = self.combine_fn(left_flat, right_flat)
            merged = merged_flat.view(B, left.shape[2], D, N).permute(0, 2, 1, 3)
            # write back
            X = X.clone()
            X[:, :, idx_r, :] = merged
        return X

    def _downsweep(self, X: Tensor, levels: int, id_val: Tensor) -> Tensor:
        B, D, P, N = *X.shape[:3], X.shape[3]
        # set last element to identity
        X = X.clone()
        X[:, :, -1, :] = id_val
        for lvl in reversed(range(levels)):
            step = 2 ** lvl
            idx_l = torch.arange(step - 1, P, 2 * step, device=X.device)
            idx_r = idx_l + step
            left = X[:, :, idx_l, :].clone()
            right = X[:, :, idx_r, :]
            Bk = left.shape[0] * left.shape[2]
            left_flat = left.permute(0, 2, 1, 3).reshape(Bk, D, N)
            right_flat = right.permute(0, 2, 1, 3).reshape(Bk, D, N)
            new_left = right
            new_right_flat = self.combine_fn(left_flat, right_flat)
            new_right = new_right_flat.view(B, left.shape[2], D, N).permute(0, 2, 1, 3)
            X = X.clone()
            X[:, :, idx_l, :] = new_left
            X[:, :, idx_r, :] = new_right
        return X
    
if __name__ == "__main__":  # simple test for additive scan
    def test_scan(inclusive: bool):
        seq = torch.arange(1, 9, dtype=torch.float32)
        B, L = 1, seq.numel()
        D, N = 1, 1
        X = seq.view(B, L, D, N).requires_grad_(True)
        
        comb = lambda u, v: u + v
        
        scan = BlellochScan(comb, inclusive=inclusive)
        out = scan(X)
        expected = torch.cumsum(X, dim=1) if inclusive else torch.cat([
            torch.zeros(B, 1, D, N), torch.cumsum(X, dim=1)[:, :-1]
        ], dim=1)
        assert torch.allclose(out, expected), "Forward mismatch"
        
        (out.sum()).backward()
        exp_grad = torch.tensor([L - i for i in range(L)], dtype=X.dtype) if inclusive else torch.tensor([L - 1 - i for i in range(L)], dtype=X.dtype)
        got_grad = X.grad.view(-1)
        assert torch.allclose(got_grad, exp_grad), "Backward mismatch"

        print(f'\nTesting Blelloch Prefix Scan: {"inclusive" if inclusive else "exclusive"}')
        print('Input:', X.flatten().detach().cpu())
        print('Output:', out.flatten().detach().cpu())
        print('Expected:', expected.flatten().detach().cpu())

    test_scan(True)
    test_scan(False)
