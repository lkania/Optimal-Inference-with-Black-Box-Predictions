# Simulation scripts for *Optimal Inference with Black-Box Predictions*

This directory contains the three Python simulation scripts used to generate the simulation figures in the paper:

- `single_prediction.py` → **Figure 3**: single prediction, varying ambient dimension.
- `aligned_predictions.py` → **Figure 4**: multiple aligned predictions, varying intrinsic dimension.
- `orthogonal_predictions.py` → **Figure 5**: multiple orthogonal predictions, varying the number of predictions.

Each script is self-contained and saves its figure as a PDF in the current working directory.

---

## 1. Create the Python environment

The scripts were run with the following relevant package versions:

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| JAX | 0.9.2 |
| jaxlib | 0.9.2 |
| NumPy | 2.4.4 |
| pandas | 3.0.3 |
| tqdm | 4.67.3 |
| SciPy | 1.17.1 |
| plotnine | 0.15.4 |
| matplotlib | 3.9.4 |

Create a Conda environment with these versions using:

```bash
conda create -n black-box-predictions -c conda-forge \
    python=3.11.15 \
    jax=0.9.2 \
    jaxlib=0.9.2 \
    numpy=2.4.4 \
    pandas=3.0.3 \
    tqdm=4.67.3 \
    scipy=1.17.1 \
    plotnine=0.15.4 \
    matplotlib=3.9.4
```

Activate the environment with:

```bash
conda activate black-box-predictions
```

---

## 2. Expected directory layout

Place the three scripts in the same working directory:

```text
.
├── single_prediction.py
├── orthogonal_predictions.py
├── aligned_predictions.py
└── README.md
```

Run all commands below from this directory.

---

## 3. Generate Figure 3: single prediction

Run:

```bash
python single_prediction.py
```

The script simulates the single-prediction setting while varying the ambient dimension.

It saves:

```text
single_prediction.pdf
```

The figure compares the following tests:

- **Chi-squared test — equation (17).** Rejects the null hypothesis whenever the squared norm of the observation is large. It ignores the prediction entirely and serves as the prediction-free benchmark.

- **Projection test — equation (25).** Projects the observation onto the span of the prediction and rejects the null hypothesis when the squared projected norm is large. With a single prediction, this is a one-dimensional chi-squared test.

- **Adaptive test — equation (27).** Combines the projection test in equation (25) with the orthogonal-complement test in equation (26), each at level `alpha / 2`, and rejects the null hypothesis if either component test rejects the null hypothesis. It therefore adapts between using the prediction span and searching outside it.

The simulation uses a fixed JAX random seed, so rerunning the script with the same software versions and parameters is reproducible.

---

## 4. Generate Figure 4: aligned predictions

Run:

```bash
python aligned_predictions.py
```

The script simulates multiple non-orthogonal predictions with controlled pairwise cosine similarity. The cosine similarity is chosen so that the squared cosine-similarity matrix has several target intrinsic dimensions.

It saves:

```text
aligned_predictions.pdf
```

The figure compares the following tests:

- **Chi-squared test — equation (17).** Rejects the null hypothesis whenever the squared norm of the observation is large. It ignores all predictions and provides the prediction-free benchmark.

- **Projection test — equation (25).** Projects the observation onto the full span of the predictions and rejects the null hypothesis when the squared projected norm is large. The test treats every orthogonal direction in the prediction span equally.

- **Intrinsic projection test — equation (31).** Uses the intrinsic projection statistic, weighting the orthogonal directions in the prediction span according to the squared singular values of the prediction matrix.

- **Adaptive test using standard projection — equation (27).** Combines the standard projection test in equation (25) with the orthogonal-complement test in equation (26), each at level `alpha / 2`, and rejects the null hypothesis if either component test rejects the null hypothesis.

- **Adaptive test using intrinsic projection — equation (32).** Combines the intrinsic projection test in equation (31) with the chi-squared test in equation (17), each at level `alpha / 2`, and rejects the null hypothesis if either component test rejects the null hypothesis.

The default script uses:

```text
d = 100
m = 50
```

and considers prediction configurations whose intrinsic dimensions are approximately:

```text
1
m / 5
m
```

The intrinsic projection test uses Monte Carlo calibration under the null, so this script is typically the most computationally expensive of the three.

---

## 5. Generate Figure 5: orthogonal predictions

Run:

```bash
python orthogonal_predictions.py
```

The script samples mutually orthogonal prediction directions and varies the number of available predictions.

It saves:

```text
orthogonal_predictions.pdf
```

The default script uses:

```text
d = 100
m in {10, 25, 50}
```

The figure compares the following tests:

- **Chi-squared test — equation (17).** Rejects the null hypothesis whenever the squared norm of the observation is large. It ignores all prediction directions and serves as the prediction-free benchmark.

- **Projection test — equation (25).** Projects the observation onto the `m`-dimensional span of the orthogonal predictions and rejects the null hypothesis when the squared projected norm is large. Under the null, the statistic has a chi-squared distribution with `m` degrees of freedom.

- **Adaptive test — equation (27).** Combines the projection test in equation (25) with the orthogonal-complement test in equation (26), each at level `alpha / 2`, and rejects the null hypothesis if either component test rejects the null hypothesis. This allows the test to use the predictions when their span is informative while retaining sensitivity to signal outside that span.
