| architecture | coupling | params | energy $W_1$ (dopri5) | $d_{ij}$ $W_1$ | path length | dopri5 NFE | energy $W_1$, Euler 4 steps | energy $W_1$, Euler 16 steps | equivariance error | train, min |
|---|---|---|---|---|---|---|---|---|---|---|
| EGNN | independent | 25510 | 0.070 | 0.008 | 2.91 | 68 | 3.51 | 0.368 | 3.7e-07 | 75 |
| EGNN | OT | 25510 | 0.075 | 0.010 | 2.91 | 56 | 1.09 | 0.253 | 2.8e-07 | 82 |
| EGNN | equivariant OT | 25510 | 0.256 | 0.013 | 2.91 | 56 | 0.90 | 0.412 | 3.1e-07 | 76 |
| MLP | independent | 25496 | 10.811 | 0.228 | 2.91 | 44 | 25.41 | 13.005 | 2.9e-01 | 23 |
| MLP | OT | 25496 | 8.276 | 0.187 | 2.90 | 32 | 10.77 | 8.898 | 3.0e-01 | 43 |
| MLP | equivariant OT | 25496 | 6.364 | 0.158 | 2.84 | 32 | 8.32 | 6.863 | 9.1e-02 | 17 |
