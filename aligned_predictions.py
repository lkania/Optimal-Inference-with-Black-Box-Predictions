# Simulation for aligned predictions and intrinsic projection in Section 6 (Figure 4).
import itertools
from functools import partial

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
    guides,
    guide_legend,
    scale_color_manual,
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


def sample_predicted_directions(
    key,
    d,
    m_predictions,
    prediction_correlation,
):
    """
    Sample m unit prediction directions with common pairwise cosine similarity rho.

    Notes
    -----
    The paper uses U_hat in R^{d x m} with prediction directions as columns,
    whereas the code stores predicted_directions = U_hat^T with shape (m, d).
    For the equicorrelation matrix Sigma_rho = (1-rho) I_m + rho 11^T, let
    Sigma_rho = L L^T and let B have orthonormal columns. The paper's
    construction is U_hat = B L^T; equivalently, the code stores
    U_hat^T = L B^T. Hence the rows have unit norm and pairwise cosine
    similarity rho, exactly as in the Section 6 simulation.
    """
    m = m_predictions
    rho = prediction_correlation

    # Sample a random m-dimensional orthonormal subspace of R^d.
    B, _ = jnp.linalg.qr(
        jax.random.normal(
            key,
            shape=(d, m),
        )
    )
    B = B[:, :m]

    # Equicorrelation Gram matrix.
    G = (
        (1.0 - rho) * jnp.eye(m)
        + rho * jnp.ones((m, m))
    )

    # G = L L^T.
    L = jnp.linalg.cholesky(G)

    # Randomly embed the correlated directions in R^d.
    predicted_directions = L @ B.T

    return predicted_directions


def orthogonal_complement_from_basis(U_tilde):
    """
    Construct an orthonormal basis for the complement of a prediction span.

    Notes
    -----
    U_tilde has orthonormal columns spanning the prediction subspace. Thus
    P = I - U_tilde U_tilde^T projects onto its orthogonal complement. The
    QR decomposition is used to retain the d - rank orthonormal directions
    in the range of P.
    """
    d, rank = U_tilde.shape

    # Projection onto the orthogonal complement.
    P = jnp.eye(d) - U_tilde @ U_tilde.T

    # Obtain orthonormal candidate directions.
    Q, _ = jnp.linalg.qr(P)

    # Retain the d-rank directions belonging to the range of P.
    norms = jnp.linalg.norm(P @ Q, axis=0)
    idx = jnp.argsort(norms)[::-1][: d - rank]

    return Q[:, idx]


def A_plus(epsilon, r, h):
    """
    Compute the positive-part lower bound A_+(epsilon, r, h) from Remark 1.

    Notes
    -----
    A(epsilon, r, h) = (epsilon^2 + h^2 - r^2) / (2h), and A_+ = max(0, A).
    For equal prediction norms and errors, A_+ is the common lower bound on
    the inner product between theta and every unit prediction direction.
    """
    return jnp.maximum(
        epsilon**2 + h**2 - r**2,
        0.0,
    ) / (2.0 * h)


def correlation_from_intrinsic_dimension(
    intrinsic_dimension,
    m_predictions,
):
    """
    Convert a target intrinsic dimension of Sigma_rho^2 into rho.

    Notes
    -----
    For the equicorrelation design in Section 6, the intrinsic dimension satisfies

        Int(Sigma_rho^2) = 1 + (m - 1) * ((1-rho) / (1 + (m-1)rho))^2.

    Solving this expression for rho yields the value returned here. The
    endpoint Int(Sigma_rho^2) = 1 would require rho = 1 and collapse the
    full-rank prediction span, so the function uses intrinsic dimensions
    strictly larger than 1.
    """
    if not 1.0 < intrinsic_dimension <= m_predictions:
        raise ValueError(
            "intrinsic_dimension must be in (1, m_predictions]. "
            "The value 1 requires rho = 1 and collapses the prediction span."
        )

    # Intermediate quantity obtained by analytically solving the
    # intrinsic-dimension expression for rho.
    t = jnp.sqrt(
        (intrinsic_dimension - 1.0)
        / (m_predictions - 1.0)
    )

    return float(
        (1.0 - t)
        / (1.0 + (m_predictions - 1.0) * t)
    )


def sample_theta_pair(
    key,
    d,
    m_predictions,
    predicted_directions,
    epsilon,
    r,
    h,
):
    """
    Sample the adversarial mean parameter theta for aligned predictions.

    Notes
    -----
    The paper stores prediction directions in U_hat in R^{d x m}; the code
    stores U_hat^T as predicted_directions. With common norm h, common error r,
    and common pairwise cosine similarity rho, the minimum-norm component in
    the prediction span is obtained by imposing the common score
    U_hat^T theta = A_+(epsilon, r, h) 1_m. In the code, G is the cosine-
    similarity matrix U_hat^T U_hat and solving G a = s constructs this
    minimum-norm span component. The remaining norm is sampled uniformly in
    the orthogonal complement. This implements the adversarial construction
    (33) used in Section 6.
    """
    key_z = key
    m = m_predictions

    U_hat = predicted_directions

    # Gram matrix of the prediction directions.
    G = U_hat @ U_hat.T

    ones = jnp.ones((m,))

    # Common desired inner product between theta and every prediction
    # direction.
    A_star = A_plus(epsilon, r, h)

    # coeffs is the desired score vector:
    #
    #     coeffs = U_hat @ theta = A_star * 1_m.
    coeffs = A_star * ones

    # These are the coordinates needed to construct the minimum-norm
    # vector in span(U_hat) producing the desired score vector.
    span_coeffs = jnp.linalg.solve(
        G,
        coeffs,
    )

    # Minimum-norm vector whose prediction scores equal coeffs.
    projection_part = U_hat.T @ span_coeffs

    projection_norm_squared = jnp.sum(projection_part**2)

    # Amount of norm still available after constructing the component
    # inside the prediction span.
    residual_radius = jnp.sqrt(
        jnp.maximum(
            epsilon**2 - projection_norm_squared,
            0.0,
        )
    )

    # Construct an orthonormal basis for the prediction span.
    U_tilde, _ = jnp.linalg.qr(U_hat.T)
    U_tilde = U_tilde[:, :m]

    complement_dim = d - m

    # Sample a uniformly random unit vector in the orthogonal complement.
    U_perp = orthogonal_complement_from_basis(U_tilde)
    z = sample_unit_sphere(key_z, complement_dim)

    # Add the residual component. Because U_perp @ z is orthogonal to
    # span(U_hat), this addition does not alter U_hat @ theta.
    theta = projection_part + residual_radius * (U_perp @ z)

    return theta


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
    predicted_directions,
    n,
    alpha,
):
    """
    Implement the prediction-free chi-squared test (17) from the paper.

    Notes
    -----
    Under H0: theta = 0, n ||X||_2^2 has a chi-squared distribution with d
    degrees of freedom. The arguments involving predictions and the PRNG key
    are retained only so that all test functions share a common interface.
    """
    d = X.shape[0]

    test_statistic = n * jnp.sum(X**2)
    critical_value = chi2.ppf(1.0 - alpha, df=d)

    return test_statistic >= critical_value


def projection_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
    m_predictions,
):
    """
    Implement the standard projection test (25).

    Notes
    -----
    QR orthonormalizes the prediction span, so the statistic depends only on
    span(U_hat), not on the pairwise alignment within that span. Under H0,
    n ||Pi_{U_hat} X||_2^2 is chi-squared with m degrees of freedom because
    the simulation uses a full-rank m-dimensional prediction span.
    """
    # Orthonormal basis for the prediction span.
    U_tilde, _ = jnp.linalg.qr(predicted_directions.T)
    U_tilde = U_tilde[:, :m_predictions]

    # Coordinates of X inside the prediction span.
    coordinates = U_tilde.T @ X

    test_statistic = n * jnp.sum(coordinates**2)

    critical_value = chi2.ppf(
        1.0 - alpha,
        df=m_predictions,
    )

    return test_statistic >= critical_value


def intrinsic_projection_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
    n_intrinsic_calibration_samples,
):
    """
    Implement the intrinsic projection test (31).

    Notes
    -----
    The paper weights the singular directions of U_hat by lambda_i^2 /
    lambda_1^2. Since predicted_directions = U_hat^T, the code statistic
    ||predicted_directions @ X||_2^2 equals lambda_1^2 times the paper's
    intrinsic-projection statistic. The same multiplicative factor appears
    in the simulated null distribution, so the resulting rejection rule is
    unchanged. The critical value is estimated by Monte Carlo draws
    Z ~ N(0, I_d).
    """
    d = X.shape[0]

    # Observed test statistic.
    test_statistic = n * jnp.sum(
        (predicted_directions @ X) ** 2
    )

    # Independent keys for null calibration draws.
    calibration_keys = jax.random.split(
        key,
        n_intrinsic_calibration_samples,
    )

    # Draw Z_1, ..., Z_B independently from N(0, I_d).
    Zs = jax.vmap(
        lambda kk: jax.random.normal(kk, shape=(d,))
    )(calibration_keys)

    # Compute the null statistic for every calibration sample.
    calibration_statistics = jnp.sum(
        (Zs @ predicted_directions.T) ** 2,
        axis=1,
    )

    # Monte Carlo estimate of the rejection threshold.
    critical_value = jnp.quantile(
        calibration_statistics,
        1.0 - alpha,
    )

    return test_statistic >= critical_value


def orthogonal_complement_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
    m_predictions,
):
    """
    Implement the orthogonal-complement test (26) for an arbitrary full-rank span.

    Notes
    -----
    QR produces an orthonormal basis for span(U_hat). The residual component
    Pi_{U_hat^perp} X has dimension d - m, so under H0 its squared norm,
    multiplied by n, is chi-squared with d - m degrees of freedom.
    """
    d = X.shape[0]

    # Orthonormal basis for the prediction span.
    U_tilde, _ = jnp.linalg.qr(predicted_directions.T)
    U_tilde = U_tilde[:, :m_predictions]

    # Projection of X onto the prediction span.
    projection_onto_span = U_tilde @ (U_tilde.T @ X)

    # Component unexplained by the prediction span.
    projection_orthogonal = X - projection_onto_span

    test_statistic = n * jnp.sum(projection_orthogonal**2)

    critical_value = chi2.ppf(
        1.0 - alpha,
        df=d - m_predictions,
    )

    return test_statistic >= critical_value


def bonferroni_projection_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
    m_predictions,
):
    """
    Implement the adaptive test based on standard projection (27).

    Notes
    -----
    The standard projection test (25) and orthogonal-complement test (26) are
    each run at level alpha / 2. This procedure adapts to unknown prediction
    quality while treating all directions in the prediction span equally.
    """
    projection_reject = projection_test(
        key=key,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha / 2,
        m_predictions=m_predictions,
    )

    orthogonal_reject = orthogonal_complement_test(
        key=key,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha / 2,
        m_predictions=m_predictions,
    )

    return jnp.logical_or(projection_reject, orthogonal_reject)


def bonferroni_intrinsic_projection_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
    n_intrinsic_calibration_samples,
):
    """
    Implement the adaptive test based on intrinsic projection (32).

    Notes
    -----
    The intrinsic projection test (31) and prediction-free chi-squared test
    (17) are each run at level alpha / 2. The procedure rejects if either
    component rejects, allowing it to exploit aligned predictions while
    retaining protection when the predictions are uninformative.
    """
    # Use independent randomness for intrinsic calibration and the
    # chi-squared component.
    key_intrinsic, key_chi_squared = jax.random.split(key)

    intrinsic_reject = intrinsic_projection_test(
        key=key_intrinsic,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha / 2,
        n_intrinsic_calibration_samples=n_intrinsic_calibration_samples,
    )

    chi_squared_reject = chi_squared_test(
        key=key_chi_squared,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha / 2,
    )

    return jnp.logical_or(
        intrinsic_reject,
        chi_squared_reject,
    )


def run_single_test(
    key,
    d,
    m_predictions,
    prediction_correlation,
    epsilon,
    r,
    h,
    n,
    alpha,
    test_fn,
):
    """
    Run one Monte Carlo trial for the aligned-prediction simulation.

    Notes
    -----
    A trial samples m prediction directions with common pairwise cosine
    similarity rho, samples the adversarial theta from construction (33),
    draws X ~ N(theta, I_d / n), applies the selected test, and returns its
    rejection indicator.
    """
    # Independent keys ensure the different random objects do not reuse
    # the same pseudo-random stream.
    key_predictions, key_theta, key_obs, key_test = jax.random.split(key, 4)

    # Generate correlated prediction directions.
    predicted_directions = sample_predicted_directions(
        key=key_predictions,
        d=d,
        m_predictions=m_predictions,
        prediction_correlation=prediction_correlation,
    )

    # Generate a true parameter consistent with the requested error level.
    theta = sample_theta_pair(
        key=key_theta,
        d=d,
        m_predictions=m_predictions,
        predicted_directions=predicted_directions,
        epsilon=epsilon,
        r=r,
        h=h,
    )

    # Sample a single observation with effective sample size n.
    X = sample_observation(
        key=key_obs,
        theta=theta,
        n=n,
    )

    # Apply the selected testing method.
    test_result = test_fn(
        key=key_test,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha,
    )

    return test_result.astype(int)


def run_tests_for_configuration(
    key,
    d,
    r,
    n_samples,
    m_predictions,
    prediction_correlation,
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
    # One independent random key per Monte Carlo trial.
    keys = jax.random.split(key, n_samples)

    # Vectorize the simulation across trials and JIT compile it.
    batched_fn = jax.jit(
        jax.vmap(
            lambda kk: run_single_test(
                key=kk,
                d=d,
                m_predictions=m_predictions,
                prediction_correlation=prediction_correlation,
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

    # Sample size and nominal type-I error level.
    n = 100
    alpha = 0.05

    # Number of null draws used to calibrate the intrinsic projection test.
    n_intrinsic_calibration_samples = 10_000

    # Ambient dimension d.
    d_values = [100]

    # Number of prediction directions m.
    m_predictions = 50

    # ------------------------------------------------------------------
    # Intrinsic-dimension configurations
    # ------------------------------------------------------------------

    # Instead of directly selecting correlations, specify three desired
    # intrinsic dimensions for the prediction geometry.
    desired_intrinsic_dimensions = [
        1.01,
        m_predictions / 5.0,
        float(m_predictions),
    ]

    # Labels for the corresponding plotting panels.
    intrinsic_dimension_panel_order = [
        r"Intrinsic dimension $\Sigma^2\approx 1$",
        r"Intrinsic dimension $\Sigma^2\approx m/5$",
        r"Intrinsic dimension $\Sigma^2=m$",
    ]

    # Convert each desired intrinsic dimension into the corresponding
    # equicorrelation rho.
    prediction_correlation_values = [
        correlation_from_intrinsic_dimension(
            intrinsic_dimension=intrinsic_dimension,
            m_predictions=m_predictions,
        )
        for intrinsic_dimension in desired_intrinsic_dimensions
    ]

    # Map each numerical rho back to its display label.
    correlation_to_intrinsic_dimension_panel = dict(
        zip(
            prediction_correlation_values,
            intrinsic_dimension_panel_order,
        )
    )

    # ------------------------------------------------------------------
    # Sanity checks for the prediction geometry
    # ------------------------------------------------------------------

    # Full-rank m-dimensional prediction spans require m <= d.
    if m_predictions > d_values[0]:
        raise ValueError("m_predictions must be at most d.")

    # The equicorrelation Gram matrix is positive definite only for
    #
    #     -1/(m-1) < rho < 1.
    for prediction_correlation in prediction_correlation_values:
        if prediction_correlation <= -1.0 / (m_predictions - 1):
            raise ValueError(
                "prediction_correlation must be greater than "
                "-1 / (m_predictions - 1)."
            )

        if prediction_correlation >= 1.0:
            raise ValueError("prediction_correlation must be less than 1.")

    # ------------------------------------------------------------------
    # Prediction-error grid
    # ------------------------------------------------------------------

    # Number of r values within each feasible interval.
    n_r_values = 30

    # At r_max, A_plus = 0.
    r_max = jnp.sqrt(epsilon**2 + h**2)

    def make_r_values(prediction_correlation):
        """
        Construct the feasible prediction-error grid for a fixed cosine similarity rho.

        Notes
        -----
        For the equicorrelation design, 1^T Sigma_rho^{-1} 1 equals
        m / (1 + (m - 1) rho). Thus the largest feasible common score is

            epsilon * sqrt((1 + (m - 1) rho) / m).

        The lower endpoint r_min solves A_+(epsilon, r_min, h) equal to this
        value, while r_max makes A_+ equal to zero, matching Section 6.
        """
        max_feasible_A = epsilon * jnp.sqrt(
            (
                1.0
                + (m_predictions - 1) * prediction_correlation
            )
            / m_predictions
        )

        # Invert A_plus(epsilon, r, h) to obtain the smallest feasible r.
        r_min = jnp.sqrt(
            jnp.maximum(
                epsilon**2
                + h**2
                - 2.0 * h * max_feasible_A,
                0.0,
            )
        )

        return jnp.linspace(
            r_min,
            r_max,
            n_r_values,
        )

    # ------------------------------------------------------------------
    # Testing procedures
    # ------------------------------------------------------------------

    # partial fixes method-specific parameters while allowing every test
    # to share the same interface inside run_single_test.
    tests = {
        "chi_squared_test": chi_squared_test,

        "projection_test": partial(
            projection_test,
            m_predictions=m_predictions,
        ),

        "bonferroni_projection_test": partial(
            bonferroni_projection_test,
            m_predictions=m_predictions,
        ),

        "intrinsic_projection_test": partial(
            intrinsic_projection_test,
            n_intrinsic_calibration_samples=n_intrinsic_calibration_samples,
        ),

        "bonferroni_intrinsic_projection_test": partial(
            bonferroni_intrinsic_projection_test,
            n_intrinsic_calibration_samples=n_intrinsic_calibration_samples,
        ),
    }

    # Store trial-level rejection indicators.
    rows = []

    # Every test is evaluated on the same number of experimental
    # configurations.
    total_per_test = (
        len(d_values)
        * len(prediction_correlation_values)
        * n_r_values
    )

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------

    # Give each test its own progress bar.
    for test_idx, (test_name, test_fn) in enumerate(tests.items()):

        # Build all combinations of dimension, prediction geometry,
        # and feasible error values for this particular test.
        test_configurations = [
            (d, prediction_correlation, r)
            for d in d_values
            for prediction_correlation
            in prediction_correlation_values
            for r in make_r_values(prediction_correlation)
        ]

        # Derive a reproducible but test-specific random stream.
        test_key = jax.random.fold_in(key, test_idx)

        # Independent key for every experimental configuration.
        outer_keys = jax.random.split(
            test_key,
            len(test_configurations),
        )

        for key_run, config in tqdm(
            zip(outer_keys, test_configurations),
            total=total_per_test,
            desc=f"Running {test_name}",
        ):
            d, prediction_correlation, r = config

            # Run all Monte Carlo trials for this configuration.
            test_results = run_tests_for_configuration(
                key=key_run,
                d=d,
                r=r,
                n_samples=n_samples,
                m_predictions=m_predictions,
                prediction_correlation=prediction_correlation,
                epsilon=epsilon,
                h=h,
                n=n,
                alpha=alpha,
                test_fn=test_fn,
            )

            # Transfer JAX results back to the host.
            test_results = jax.device_get(test_results)

            # Store the rejection indicator from each trial.
            for result in test_results:
                rows.append({
                    "test": test_name,
                    "d": int(d),
                    "r": float(r),
                    "epsilon": float(epsilon),
                    "h": float(h),
                    "n": int(n),
                    "m_predictions": int(m_predictions),
                    "prediction_correlation": float(prediction_correlation),
                    "test_result": int(result),
                })

    # ------------------------------------------------------------------
    # Summarize Monte Carlo results
    # ------------------------------------------------------------------

    df = pd.DataFrame(rows)

    # Since test_result is binary, its mean estimates power under the simulated alternative.
    summary_df = (
        df.groupby(
            [
                "test",
                "d",
                "r",
                "m_predictions",
                "prediction_correlation",
            ]
        )["test_result"]
        .mean()
        .reset_index()
        .rename(columns={"test_result": "rejection_rate"})
    )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    # Work on a copy so that display-specific labels do not modify the
    # underlying simulation summary.
    plot_df = summary_df.copy()

    # Map internal test names to the labels used in the paper figures.
    test_name_map = {
        "chi_squared_test": "Chi-squared",
        "projection_test": "Projection",
        "intrinsic_projection_test": "Intrinsic projection",
        "bonferroni_intrinsic_projection_test": "Adaptive (Intrinsic projection)",
        "bonferroni_projection_test": "Adaptive (Projection)",
    }

    plot_df["test"] = plot_df["test"].replace(test_name_map)

    # Explicit test ordering controls both line/legend ordering.
    test_order = [
        "Chi-squared",
        "Projection",
        "Intrinsic projection",
        "Adaptive (Intrinsic projection)",
        "Adaptive (Projection)",
    ]

    # Assign a distinct color to each testing procedure.
    test_colors = {
        "Chi-squared": "#00BA38",  # green
        "Projection": "#619CFF",  # blue
        "Intrinsic projection": "#984EA3",  # purple
        "Adaptive (Intrinsic projection)": "#F8766D",  # red
        "Adaptive (Projection)": "#E69F00",  # orange
    }

    plot_df["test"] = pd.Categorical(
        plot_df["test"],
        categories=test_order,
        ordered=True,
    )

    # Convert numerical correlation values into the intended
    # intrinsic-dimension panel labels.
    plot_df["intrinsic_dimension_panel"] = pd.Categorical(
        plot_df["prediction_correlation"].map(
            correlation_to_intrinsic_dimension_panel
        ),
        categories=intrinsic_dimension_panel_order,
        ordered=True,
    )

    # Plot estimated power as a function of prediction error.
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

            # Each panel corresponds to a different intrinsic dimension.
            # free_x is used because each rho produces a different
            # feasible r interval.
            + facet_wrap(
        "~ intrinsic_dimension_panel",
        scales="free_x",
    )
            + theme_bw()
            + labs(
        title="Power across intrinsic dimensions (d=100,m=50)",
        x="Unknown predictions' error (larger is worse)",
        y="Power (higher is better)",
        color="Test",
    )

            # Apply the manually specified test colors.
            + scale_color_manual(
        values=test_colors,
        breaks=test_order,
    )

            # Arrange the legend over two rows.
            + guides(
        color=guide_legend(
            nrow=2,
            byrow=True,
        )
    )
            + theme(
        # Place legend above the panels.
        legend_position="top",
        legend_direction="horizontal",
        legend_box="vertical",

        # Compact legend text and spacing.
        legend_text=element_text(size=8),
        legend_title=element_text(size=9, margin={"r": 4}),
        legend_margin=0,
        legend_box_margin=0,

        # Plot/facet/axis formatting.
        plot_title=element_text(margin={"b": 2}),
        strip_text=element_text(size=9),
        axis_title_x=element_text(size=10),
        axis_title_y=element_text(size=10),
        axis_text_x=element_text(size=8),
        axis_text_y=element_text(size=8),
        panel_spacing_x=0.02,
    )
    )

    # Save the figure.
    p_r.save(
        "aligned_predictions.pdf",
        dpi=300,
        width=8,
        height=4,
    )