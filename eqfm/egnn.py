"""E(n)-equivariant GNN vector field for Equivariant Flow Matching (Klein et al. 2023, eq. 17-20)."""

import torch
import torch.nn as nn


class MLPSequence(nn.Module):
    """
    Sequence of MLPs with a fixed hidden dim
    """
    
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, activation=nn.SiLU, closing_activation=None):
        super(MLPSequence, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        
        self.activation = activation
        self.n_layers = n_layers
        
        self.model = nn.Sequential()
        
        self.model.append(nn.Linear(in_dim, hid_dim))
        self.model.append(activation())
        
        if n_layers > 2:
            for i in range(n_layers - 2):
                self.model.append(nn.Linear(hid_dim, hid_dim))
                self.model.append(activation())
                
        self.model.append(nn.Linear(hid_dim, out_dim))
        
        if closing_activation is not None:
            self.model.append(closing_activation())
        
    def forward(self, x):
        x = self.model(x)
        return x


class EGNNLayer(nn.Module):
    """A single E(n)-equivariant GNN layer on the fully connected graph (eq. 17-19)."""

    def __init__(self, hidden_dim: int, message_dim: None | int = None):
        """
        hidden_dim:  dimension H of the invariant features h.
        message_dim: dimension of the messages m_ij and hidden width of all MLPs
                     phi_e, phi_d, phi_m, phi_h (defaults to hidden_dim).
        """
        super().__init__()
        self.hid_dim = hidden_dim
        self.message_dim = self.hid_dim if not message_dim else message_dim
        
        # phi_e(h_i, h_j, d_{i,j}) computes message
        self.phi_e = MLPSequence(
            in_dim=2 * self.hid_dim + 1,
            hid_dim=self.message_dim,
            out_dim=self.message_dim,
            n_layers=2
        )
        
        # phi_d(m_{i,j}) \in \mathbb{R} computes weight for (x_i - x_j)
        self.phi_d = MLPSequence(
            in_dim=self.message_dim,
            hid_dim=self.message_dim,
            out_dim=1,
            n_layers=2
        )
        
        # phi_m(m_{i, j}) \in \mathbb{R} computes weight for the message m_{i, j}
        self.phi_m = MLPSequence(
            in_dim=self.message_dim,
            hid_dim=self.message_dim,
            out_dim=1,
            n_layers=2,
            closing_activation=nn.Sigmoid
        ) 
        
        # phi_h(h_i, m_i) computes new hidden representation h_i^{l+1}
        self.phi_h = MLPSequence(
            in_dim=self.hid_dim + self.message_dim,
            hid_dim=self.message_dim,
            out_dim=self.hid_dim,
            n_layers=2
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, N, H] invariant particle features at layer l
        x: [B, N, D] particle coordinates at layer l

        returns:
            h_new: [B, N, H] features at layer l+1
            x_new: [B, N, D] coordinates at layer l+1
        """
        N = h.size(1)
        
        
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)                                  # [B, N, N, D]
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)                                  # [B, N, N, D]
        
        x_dif = x_i - x_j                                                           # [B, N, N, D]
        
        d_sq = (x_dif ** 2).sum(-1)                                                 # [B, N, N]
        d = (d_sq + 1e-9) ** 0.5                                                    # [B, N, N]
        
        x_dif_normalized = x_dif / (d.unsqueeze(-1) + 1)                            # [B, N, N, D]
        
        # h ~ [B, N, H]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)                                  # [B, N, 1, H] --> [B, N, N, H]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)                                  # [B, 1, N, H] --> [B, N, N, H]
        h_cat = torch.cat([h_i, h_j], dim=-1)                                       # [B, N, N, 2H]
        
        
        m_ij = self.phi_e(torch.cat([h_cat, d_sq.unsqueeze(-1)], dim=-1))           # [B, N, N, H]
        
        # weights proportional to the importance of the j-th coordinate to the i-th node, computed based on the message passed:
        coord_coef = self.phi_d(m_ij)                                               # [B, N, N, 1]
        
        x_new = x + (x_dif_normalized * coord_coef).sum(dim=2)                      # [B, N, D]
        
        diag = torch.eye(N, N, dtype=m_ij.dtype, device=m_ij.device).unsqueeze(0).unsqueeze(-1)
        mask = torch.ones((1, N, N, 1), dtype=m_ij.dtype, device=m_ij.device) - diag
        # mask ~ [1, N, N, 1]
        
        m_ij_masked = m_ij * mask
        
        message_coef = self.phi_m(m_ij_masked)                                      # [B, N, N, 1]
        
        m_i = (message_coef * m_ij_masked).sum(dim=2)                               # [B, N, H]
        
        h_new = h + self.phi_h(torch.cat([h, m_i], dim=-1))
        
        return h_new, x_new


class EGNNVectorField(nn.Module):
    """v_theta(t, x): [B], [B, N, D] -> [B, N, D], equivariant to S(N) x O(D), preserves the center of mass."""

    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        self.emb = nn.Linear(1, hidden_dim)
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.layers.append(EGNNLayer(hidden_dim))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: [B] or [B, 1] — time
        x: [B, N, D] — configuration (centered)

        returns: v [B, N, D]
        """
        N = x.size(1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        # t ~ [B, 1]
        
        h_i = self.emb(t).unsqueeze(1).expand(-1, N, -1)            # [B, N, H]
        
        x_i = x
        
        for layer in self.layers:
            h_i, x_i = layer(h_i, x_i)
            
        v = x_i - x
        v = v - v.mean(dim=1, keepdim=True)
        return v
