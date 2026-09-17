# Simulation for the single-prediction setting in Section 6 (Figure 3).
import itertools

import jax
import jax.numpy as jnp
import pandas as pd

from tqdm import tqdm
from scipy.stats import chi2

from plotnine import (
    ggplot,
    aes,
    element_text,
    geom_line,
    geom_point,
    facet_wrap,
    theme_bw,
    theme,
    labs,
)


def sample_unit_sphere(key, d):
    """
    Sample a direction uniformly from the unit sphere in R^d.

    Notes
    -----
    If Z ~ N(0, I_d), rotational invariance implies that Z / ||Z||_2 is
    uniform on the unit sphere. This is the sphere notation used throughout
    the paper when sampling random directions.
    """
    z = jax.random.normal(key, shape=(d,))
    return z / jnp.linalg.norm(z)


def orthogonal_complement_basis(u_hat):
    """
    Construct an orthonormal basis for the complement of one unit direction.

    Notes
    -----
    For a unit vector u_hat, P = I - u_hat u_hat^T is the orthogonal
    projection onto span(u_hat)^perp. A QR decomposition of P is used to
    extract d - 1 orthonormal directions in that complement.
    """
    d = u_hat.shape[0]

    P = jnp.eye(d) - jnp.outer(u_hat, u_hat)
    Q, _ = jnp.linalg.qr(P)

    norms = jnp.linalg.norm(P @ Q, axis=0)
    idx = jnp.argsort(norms)[::-1][: d - 1]

    return Q[:, idx]


def A_plus(epsilon, r, h):
    """
    Compute the positive-part geometric lower bound A_+(epsilon, r, h).

    Notes
    -----
    Remark 1 of the paper defines A(epsilon, r, h) =
    (epsilon^2 + h^2 - r^2) / (2h) and A_+ = max(0, A). For a unit
    prediction direction u_hat, A_+ is the smallest feasible magnitude of
    the projection of theta onto u_hat under the localized alternative.
    """
    return jnp.maximum(
        epsilon**2 + h**2 - r**2,
        0.0,
    ) / (2.0 * h)


def sample_theta_pair(
    key,
    d,
    epsilon,
    r,
    h,
):
    """
    Sample the prediction and an adversarial mean parameter theta.

    Notes
    -----
    This is the single-prediction version of the simulation construction in
    Section 6. First sample a unit prediction direction u_hat and set
    prediction = h u_hat. The component of theta along u_hat is A_+, and the
    remaining norm is placed uniformly in span(u_hat)^perp. The resulting
    theta has norm epsilon and, on the feasible grid, prediction error r.
    """
    key_u, key_z = jax.random.split(key)

    # Prediction direction.
    u_hat = sample_unit_sphere(key_u, d)

    # Prediction vector with norm h.
    theta_hat = h * u_hat

    # Component of theta along the prediction direction.
    A = A_plus(epsilon, r, h)
    A = jnp.minimum(A, epsilon)

    # Basis for the orthogonal complement of u_hat.
    U_perp = orthogonal_complement_basis(u_hat)

    # Random unit vector in the orthogonal complement.
    z = sample_unit_sphere(key_z, d - 1)

    theta = (
        A * u_hat
        + jnp.sqrt(epsilon**2 - A**2) * (U_perp @ z)
    )

    return theta_hat, theta


def sample_observation(key, theta, n):
    """
    Sample the Gaussian sequence-model observation X ~ N(theta, I_d / n).

    Notes
    -----
    The paper uses X for the sample mean of n observations, so the simulation
    works directly with its equivalent distribution N(theta, I_d / n).
    """
    d = theta.shape[0]
    noise = jax.random.normal(key, shape=(d,))

    return theta + noise / jnp.sqrt(n)


def chi_squared_test(
    key,
    X,
    prediction,
    n,
    alpha,
):
    """
    Implement the prediction-free chi-squared test (21) from the paper.

    Notes
    -----
    Under H0: theta = 0, n ||X||_2^2 has a chi-squared distribution with d
    degrees of freedom. The arguments involving predictions and the PRNG key
    are retained only so that all test functions share a common interface.
    """
    d = X.shape[0]

    test_statistic = n * jnp.sum(X**2)

    critical_value = chi2.ppf(
        1.0 - alpha,
        df=d,
    )

    return test_statistic >= critical_value


def projection_test(
    key,
    X,
    prediction,
    n,
    alpha,
):
    """
    Implement the projection test (34) specialized to one prediction.

    Notes
    -----
    The statistic is n <u_hat, X>^2, where u_hat is the normalized prediction
    direction. Under H0 it has a chi-squared distribution with one degree of
    freedom. This test always uses the prediction.
    """
    normalized_prediction = (
        prediction / jnp.linalg.norm(prediction)
    )

    test_statistic = (
        n
        * jnp.dot(normalized_prediction, X) ** 2
    )

    critical_value = chi2.ppf(
        1.0 - alpha,
        df=1,
    )

    return test_statistic >= critical_value


def orthogonal_complement_test(
    key,
    X,
    prediction,
    n,
    alpha,
):
    """
    Implement the orthogonal-complement test (35) for one prediction.

    Notes
    -----
    The statistic is n ||Pi_{u_hat^perp} X||_2^2. Under H0 it has a
    chi-squared distribution with d - 1 degrees of freedom.
    """
    d = X.shape[0]

    normalized_prediction = (
        prediction / jnp.linalg.norm(prediction)
    )

    projection_onto_prediction = (
        jnp.dot(X, normalized_prediction)
        * normalized_prediction
    )

    projection_orthogonal = (
        X - projection_onto_prediction
    )

    test_statistic = (
        n
        * jnp.sum(projection_orthogonal**2)
    )

    critical_value = chi2.ppf(
        1.0 - alpha,
        df=d - 1,
    )

    return test_statistic >= critical_value


def bonferroni_projection_test(
    key,
    X,
    prediction,
    n,
    alpha,
):
    """
    Implement the adaptive test (36) specialized to one prediction.

    Notes
    -----
    The test applies the projection test and the orthogonal-complement test
    at level alpha / 2 and rejects if either component rejects. This is the
    adaptive procedure compared with the other tests in Figure 3.
    """
    projection_reject = projection_test(
        key=key,
        X=X,
        prediction=prediction,
        n=n,
        alpha=alpha / 2,
    )

    orthogonal_reject = orthogonal_complement_test(
        key=key,
        X=X,
        prediction=prediction,
        n=n,
        alpha=alpha / 2,
    )

    return jnp.logical_or(
        projection_reject,
        orthogonal_reject,
    )


def run_single_test(
    key,
    d,
    epsilon,
    r,
    h,
    n,
    alpha,
    test_fn,
):
    """
    Run one Monte Carlo trial for the single-prediction simulation.

    Notes
    -----
    A trial samples the prediction and adversarial theta, draws
    X ~ N(theta, I_d / n), applies the selected test, and returns its rejection
    indicator.
    """
    key_theta_pair, key_obs, key_test = jax.random.split(
        key,
        3,
    )

    prediction, theta = sample_theta_pair(
        key=key_theta_pair,
        d=d,
        epsilon=epsilon,
        r=r,
        h=h,
    )

    # A single observation with total effective sample size n.
    X = sample_observation(
        key=key_obs,
        theta=theta,
        n=n,
    )

    test_result = test_fn(
        key=key_test,
        X=X,
        prediction=prediction,
        n=n,
        alpha=alpha,
    )

    return test_result.astype(int)


def run_tests_for_configuration(
    key,
    d,
    r,
    n_samples,
    epsilon,
    h,
    n,
    alpha,
    test_fn,
):
    """
    Run all Monte Carlo trials for one fixed simulation configuration.

    Notes
    -----
    The function splits the PRNG key into independent trial keys, vectorizes
    run_single_test with jax.vmap, and JIT-compiles the resulting batch.
    """
    keys = jax.random.split(
        key,
        n_samples,
    )

    batched_fn = jax.jit(
        jax.vmap(
            lambda kk: run_single_test(
                key=kk,
                d=d,
                epsilon=epsilon,
                r=r,
                h=h,
                n=n,
                alpha=alpha,
                test_fn=test_fn,
            )
        )
    )

    return batched_fn(keys)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------

    # Master PRNG key for reproducibility.
    key = jax.random.PRNGKey(0)

    # Number of independent Monte Carlo trials per configuration.
    n_samples = 1000

    # Separation radius; simulated alternatives satisfy ||theta||_2 = epsilon.
    epsilon = 0.3

    # Common prediction norm h.
    h = epsilon

    # Sample size in X ~ N(theta, I_d / n).
    n = 100

    # Nominal type-I error level.
    alpha = 0.05

    # Ambient dimensions considered in the simulation.
    d_values = [2, 10, 100]

    # ------------------------------------------------------------------
    # Prediction-error grid
    # ------------------------------------------------------------------

    # Prediction-error grid r.
    r_values = jnp.linspace(
        0.0,
        jnp.sqrt(epsilon**2 + h**2),
        30,
    )

    # ------------------------------------------------------------------
    # Testing procedures
    # ------------------------------------------------------------------

    # Testing procedures compared in the figure.
    tests = {
        "chi_squared_test": chi_squared_test,
        "projection_test": projection_test,
        "bonferroni_projection_test": bonferroni_projection_test,
    }

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------

    # All combinations of test, dimension, and prediction error.
    configurations = list(
        itertools.product(
            tests.items(),
            d_values,
            r_values,
        )
    )

    outer_keys = jax.random.split(
        key,
        len(configurations),
    )

    rows = []

    for key_run, config in tqdm(
        zip(
            outer_keys,
            configurations,
        ),
        total=len(configurations),
        desc="Running experiments",
    ):
        (test_name, test_fn), d, r = config

        test_results = run_tests_for_configuration(
            key=key_run,
            d=d,
            r=r,
            n_samples=n_samples,
            epsilon=epsilon,
            h=h,
            n=n,
            alpha=alpha,
            test_fn=test_fn,
        )

        test_results = jax.device_get(
            test_results
        )

        # Alignment induced by the geometric construction.
        A = A_plus(
            epsilon,
            r,
            h,
        )

        A = jnp.minimum(
            A,
            epsilon,
        )

        A_plus_squared_over_epsilon_squared = (
            A**2 / epsilon**2
        )

        for result in test_results:
            rows.append({
                "test": test_name,
                "d": int(d),
                "r": float(r),
                "epsilon": float(epsilon),
                "h": float(h),
                "n": int(n),
                "A_plus_squared_over_epsilon_squared": float(
                    A_plus_squared_over_epsilon_squared
                ),
                "test_result": int(result),
            })

    # ------------------------------------------------------------------
    # Summarize Monte Carlo results
    # ------------------------------------------------------------------

    df = pd.DataFrame(rows)

    # Estimate power by averaging the rejection indicators.
    summary_df = (
        df.groupby(
            [
                "test",
                "d",
                "r",
                "A_plus_squared_over_epsilon_squared",
            ]
        )["test_result"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "test_result": "rejection_rate"
            }
        )
    )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    plot_df = summary_df.copy()

    test_name_map = {
        "chi_squared_test": "Chi-squared",
        "projection_test": "Projection",
        "bonferroni_projection_test": "Adaptive",
    }

    plot_df["test"] = (
        plot_df["test"]
        .replace(test_name_map)
    )

    # Facet labels ordered by increasing dimension.
    dimension_order = [
        f"Ambient dimension d = {d}"
        for d in sorted(
            plot_df["d"].unique()
        )
    ]

    plot_df["ambient_dimension"] = pd.Categorical(
        "Ambient dimension d = "
        + plot_df["d"].astype(str),
        categories=dimension_order,
        ordered=True,
    )

    p_r = (
        ggplot(
            plot_df,
            aes(
                x="r",
                y="rejection_rate",
                color="test",
                group="test",
            ),
        )
        + geom_line()
        + geom_point(size=2)
        + facet_wrap(
            "~ ambient_dimension"
        )
        + theme_bw()
        + labs(
            title="Power across ambient dimensions",
            x="Unknown prediction's error (larger is worse)",
            y="Power (higher is better)",
            color="Test",
        )
        + theme(
            legend_position="top",

            # Shrink spaces around legend.
            legend_margin=0,
            legend_box_margin=0,

            # Reduce facet spacing.
            panel_spacing_x=0.01,

            # Reduce title spacing.
            plot_title=element_text(
                margin={"b": 2}
            ),

            # Reduce legend spacing.
            legend_title=element_text(
                margin={"r": 4}
            ),
        )
    )

    p_r.save(
        "single_prediction.pdf",
        dpi=300,
        width=8,
        height=4,
    )