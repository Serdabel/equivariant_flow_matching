# Symmetry in the model vs. symmetry in the coupling: equivariant flow matching on DW4

A from-scratch reimplementation and ablation of **Equivariant Flow Matching** (Klein, Krämer, Noé, NeurIPS 2023)
on the DW4 many-body system (4 particles in 2D, double-well pair potential).

**Question.** For generative models of particle systems, where should the symmetry go?
* into the **architecture**: an E(n)-equivariant GNN (EGNN) vs. a plain MLP of matched size (~25k parameters);
* into the **coupling** of noise and data in flow matching: independent pairs vs. minibatch OT vs. equivariant OT
  (pairs aligned over permutations and rotations before matching).

All 2 × 3 combinations are trained with the same loss, data and schedule, and compared on sample quality,
transport path length, integration cost (NFE) and equivariance of the learned flow.

## Results

*To be filled from `results/summary.md` after the full run.* Figures: `results/fig_distributions.png`,
`results/fig_nfe.png`.

## Repository

```
eqfm/                 library
  egnn.py             E(n)-equivariant GNN vector field (eq. 17-20 of the paper)
  mlp.py              non-equivariant MLP baseline
  flow_matching.py    mean-free prior, conditional flow matching loss, training loop, ODE sampling
  couplings.py        minibatch OT; Hungarian + Kabsch alignment; equivariant OT
  dw4.py              DW4 data loading and energy
  plotting.py
scripts/
  prepare_dw4.py      download the authors' DW4 data (OSF) and convert it to plain arrays
  precompute_eq_ot.py equivariant-OT pairs, computed once before training (as in App. C.4 of the paper)
  run_experiment.py   train + evaluate one configuration
  make_figures.py     figures and summary table
tests/                equivariance tests of the EGNN; tests of OT, Kabsch, alignment and equivariant OT
run_all.sh            the full experiment (6 configurations in parallel on CPU)
```

## Reproduce

```bash
pip install -r requirements.txt
pytest                                               # 12 tests
nohup bash run_all.sh > results/run_all.log 2>&1 &   # ~2 h on a 20-thread CPU
```

## Implementation notes

* The training set is centered: the prior and the model live on the zero-center-of-mass subspace.
* Equivariant OT costs ~4–10 s per batch of 256 with the per-pair loop, so pairs are precomputed
  (10 pair sets, one per epoch). For DW4 the per-pair Hungarian step is replaced by an exact search over all
  4! permutations (`eq_ot_coupling_bruteforce`), tested to give the same result as the loop version.
* `torchdyn` 1.0.6 `rk4` evaluates time-dependent vector fields at shifted stage times
  (`c[0..2]` instead of `c[1..3]`); `torchdiffeq` is used instead.

## Scope and next steps

DW4 only, one seed. Not done yet: LJ13 / LJ55, likelihoods and importance-sampling ESS (Boltzmann generator
evaluation), several seeds.

## Authorship

The core of the project was implemented by me (Sergei Usanov): the EGNN layer and vector field, the flow matching
loss and prior, the DW4 energy, the training loop, the ODE sampling, minibatch OT, the Hungarian/Kabsch alignment
and equivariant OT. I used Claude (Anthropic) as a coding assistant for code review and debugging and assisting with
routine implementation, testing, data processing, visualization, documentation and this README.

## Reference

L. Klein, A. Krämer, F. Noé. *Equivariant flow matching.* NeurIPS 2023. arXiv:2306.15030.
