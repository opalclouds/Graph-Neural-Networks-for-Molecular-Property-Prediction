# Graph Neural Networks for Molecular Property Prediction

Do graph neural networks beat classical fingerprint baselines at predicting
blood-brain barrier permeability? This repository benchmarks a GCN and a GIN
against a random forest on Morgan fingerprints, under a scaffold split, with
results reported across five seeds.

**Short answer:** not significantly. With proper atom featurization the GIN
reaches parity with the fingerprint baseline, but the difference is not
statistically significant.

## Results

| Model | Test ROC-AUC | Features |
|---|---|---|
| Random Forest | 0.6924 ± 0.0083 | Morgan fingerprints (2048-bit, radius 2) |
| GCN | 0.6259 ± 0.0076 | 7 raw atom descriptors |
| **GIN** | **0.7126 ± 0.0156** | 29-dim one-hot atom features |

Mean ± standard deviation over 5 random seeds.

GIN vs. Random Forest: difference = +0.0202, p = 0.051 (two-sample t-test).
**Not significant at α = 0.05.**

## Dataset

[BBBP](https://moleculenet.org/datasets-1) from MoleculeNet: 2,050 compounds
labelled for blood-brain barrier penetration. 11 molecules fail RDKit parsing
and are dropped, leaving 2,039. The dataset is imbalanced at 76.5% positive,
which is why ROC-AUC is used rather than accuracy, and why the loss applies
class weighting.

## Scaffold splitting

Molecules are grouped by Bemis-Murcko scaffold (the core ring system, stripped
of side chains) and whole groups are assigned to a single split, so no scaffold
appears in both training and test data. The 2,039 molecules span 1,025 distinct
scaffolds.

| Split | Molecules | Positive rate |
|---|---|---|
| Train | 1,631 | 0.822 |
| Valid | 203 | 0.552 |
| Test | 205 | 0.522 |

This is deliberately harder than a random split, which would let near-identical
molecules appear on both sides and inflate scores to ~0.95. It also creates a
substantial distribution shift: the model trains on mostly brain-penetrant
compounds and is tested on a near-balanced set of unfamiliar scaffolds.

## Why the GCN underperformed

The initial GCN fed `GetAtomicNum()` as a raw integer, implying a false ordinal
relationship between elements (carbon = 6, nitrogen = 7). Replacing this with
one-hot encoding over 9 common elements, plus one-hot degree, hybridization and
hydrogen count, raised test ROC-AUC from 0.626 to 0.713. Switching from GCN to
GIN, which is provably more expressive at distinguishing graph structures, and
from mean to sum pooling contributed to the same jump.

Both graph models show a large validation-test gap (GIN: 0.96 validation vs.
0.71 test), reflecting the scaffold shift rather than ordinary overfitting.

## Protocol

- Test scores are read only at the epoch with the best validation ROC-AUC; the
  test set never influences training or model selection.
- Class imbalance handled via `pos_weight` in `BCEWithLogitsLoss`.
- Five seeds per model; no seed selection or additional runs after seeing results.

## Running it

```bash
pip install rdkit torch_geometric scikit-learn pandas
```

Open `gnn_bbbp.ipynb` and run all cells. Training takes roughly five minutes
per model on a single T4 GPU.

## Limitations

- 205 test molecules is small; the ±0.016 standard deviation on the GIN reflects
  that, and conclusions about a 0.02 difference are correspondingly weak.
- Bond features are unused. Single and aromatic bonds are currently
  indistinguishable to the model, and edge features would be the natural next step.
- Only one dataset. A conclusion about graph models versus fingerprints in
  general would require testing across several MoleculeNet tasks.
- No hyperparameter search was run for either model beyond the architectural
  changes described above.

## References

- Wu et al., *MoleculeNet: A Benchmark for Molecular Machine Learning* (2018)
- Xu et al., *How Powerful are Graph Neural Networks?* (ICLR 2019) — GIN
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional
  Networks* (ICLR 2017) — GCN
