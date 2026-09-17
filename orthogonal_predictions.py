# Simulation for orthogonal predictions in Section 6 (Figure 5).
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


def sample_predicted_directions(key, d, m_predictions):
    """
    Sample m mutually orthogonal prediction directions in R^d.

    Notes
    -----
    The paper writes U_hat in R^{d x m} with prediction directions as columns.
    The code stores its transpose: predicted_directions = U_hat^T has shape
    (m, d), with one direction per row. QR applied to a Gaussian d x m matrix
    produces a random orthonormal system, so predicted_directions @
    predicted_directions.T = I_m. This is the rho = 0 setting in Section 6.
    """

    # Random Gaussian d x m matrix.
    Z = jax.random.normal(
        key,
        shape=(d, m_predictions),
    )

    # QR gives orthonormal columns in Q.
    Q, _ = jnp.linalg.qr(Z)

    # Store the m orthonormal prediction directions as rows.
    predicted_directions = Q[:, :m_predictions].T

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

    # Projection matrix onto span(U_tilde)^\perp.
    P = jnp.eye(d) - U_tilde @ U_tilde.T

    # Obtain orthonormal candidate directions.
    Q, _ = jnp.linalg.qr(P)

    # Identify the d-rank directions corresponding to the range of P.
    norms = jnp.linalg.norm(P @ Q, axis=0)
    idx = jnp.argsort(norms)[::-1][: d - rank]

    return Q[:, idx]


def A_plus(epsilon, r, h):
    """
    Compute A(epsilon, r, h) from Remark 1 of the paper.

    Notes
    -----
    A(epsilon, r, h) = (epsilon^2 + h^2 - r^2) / (2h). The simulation grid
    is restricted to the feasible regime where A is nonnegative, so A equals
    the positive-part quantity A_+ used in the paper.
    """
    return (epsilon**2 + h**2 - r**2) / (2.0 * h)


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
    Sample the adversarial mean parameter theta for orthogonal predictions.

    Notes
    -----
    The code stores U_hat^T as predicted_directions. Under orthogonality,
    the paper shows that A^*(epsilon, r) = A_+(epsilon, r). With equal errors
    and norms, theta has coordinate A_+ along each of the m prediction
    directions. The remaining norm is placed uniformly in the orthogonal
    complement, matching the simulation construction (52). The r grid is
    chosen so that m A_+^2 <= epsilon^2.
    """

    # Randomness is needed only for the direction in the orthogonal complement.
    key_z = key

    # Since the rows of predicted_directions are orthonormal,
    # its transpose is an orthonormal basis for their span.
    U_tilde = predicted_directions.T

    # Common coordinate of theta along every prediction direction.
    A_star = A_plus(epsilon, r, h)

    # A is the m-dimensional vector
    #
    #     (A_star, ..., A_star).
    A = jnp.full((m_predictions,), A_star)

    # Component of theta lying inside the span of the predictions.
    projection_part = U_tilde @ A

    # Because U_tilde has orthonormal columns, this also equals sum(A^2).
    projection_norm_squared = jnp.sum(A**2)

    # Remaining radius that must be placed outside the prediction span
    # so that the final theta has norm epsilon.
    residual_radius = jnp.sqrt(
        jnp.maximum(
            epsilon**2 - projection_norm_squared,
            0.0,
        )
    )

    # Dimension of the orthogonal complement.
    complement_dim = d - m_predictions

    # Basis for directions orthogonal to all predictions.
    U_perp = orthogonal_complement_from_basis(U_tilde)

    # Uniformly random unit vector in that complement.
    z = sample_unit_sphere(key_z, complement_dim)

    # Combine the prediction-span component with the orthogonal residual.
    #
    # The two terms are orthogonal, so their squared norms add.
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

    # Standard Gaussian noise.
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
    Implement the prediction-free chi-squared test (21) from the paper.

    Notes
    -----
    Under H0: theta = 0, n ||X||_2^2 has a chi-squared distribution with d
    degrees of freedom. The arguments involving predictions and the PRNG key
    are retained only so that all test functions share a common interface.
    """
    d = X.shape[0]

    # Global squared Euclidean norm of the observation.
    test_statistic = n * jnp.sum(X**2)

    # Null critical value.
    critical_value = chi2.ppf(1.0 - alpha, df=d)

    return test_statistic >= critical_value


def projection_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
):
    """
    Implement the standard projection test (34) for orthogonal predictions.

    Notes
    -----
    Because the rows of predicted_directions are orthonormal,
    predicted_directions @ X gives the coordinates of X in the prediction
    span. Under H0, n ||Pi_{U_hat} X||_2^2 is chi-squared with m degrees of
    freedom. This test always uses all prediction directions.
    """
    m = predicted_directions.shape[0]

    # Coordinates of X in the prediction basis.
    coordinates = predicted_directions @ X

    # Squared norm of the projected component.
    test_statistic = n * jnp.sum(coordinates**2)

    # The prediction span has dimension m.
    critical_value = chi2.ppf(
        1.0 - alpha,
        df=m,
    )

    return test_statistic >= critical_value


def orthogonal_complement_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
):
    """
    Implement the orthogonal-complement test (35).

    Notes
    -----
    The statistic is n ||Pi_{U_hat^perp} X||_2^2. For m orthogonal prediction
    directions in R^d, the complement has dimension d - m, so the null
    statistic is chi-squared with d - m degrees of freedom.
    """
    d = X.shape[0]
    m = predicted_directions.shape[0]

    # Orthogonal projection of X onto the span of the predictions.
    projection_onto_span = (
        predicted_directions.T
        @ (predicted_directions @ X)
    )

    # Residual component orthogonal to every prediction direction.
    projection_orthogonal = X - projection_onto_span

    test_statistic = n * jnp.sum(projection_orthogonal**2)

    # Orthogonal complement has dimension d-m.
    critical_value = chi2.ppf(
        1.0 - alpha,
        df=d - m,
    )

    return test_statistic >= critical_value


def bonferroni_projection_test(
    key,
    X,
    predicted_directions,
    n,
    alpha,
):
    """
    Implement the adaptive projection test (36).

    Notes
    -----
    The projection test (34) and orthogonal-complement test (35) are each run
    at level alpha / 2. The procedure rejects if either component rejects.
    This is the adaptive test studied for orthogonal predictions in Section 4
    and Figure 5.
    """
    projection_reject = projection_test(
        key=key,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha / 2,
    )

    orthogonal_reject = orthogonal_complement_test(
        key=key,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha / 2,
    )

    return jnp.logical_or(projection_reject, orthogonal_reject)


def run_single_test(
    key,
    d,
    m_predictions,
    epsilon,
    r,
    h,
    n,
    alpha,
    test_fn,
):
    """
    Run one Monte Carlo trial for the orthogonal-prediction simulation.

    Notes
    -----
    A trial samples m orthogonal prediction directions, samples the adversarial
    theta from construction (52), draws X ~ N(theta, I_d / n), applies the
    selected test, and returns its rejection indicator.
    """

    # Independent random streams for:
    #   prediction directions,
    #   theta,
    #   observation,
    #   and the test.
    key_predictions, key_theta, key_obs, key_test = jax.random.split(key, 4)

    # Sample m orthogonal prediction directions.
    predicted_directions = sample_predicted_directions(
        key=key_predictions,
        d=d,
        m_predictions=m_predictions,
    )

    # Sample theta with the requested common prediction error r.
    theta = sample_theta_pair(
        key=key_theta,
        d=d,
        m_predictions=m_predictions,
        predicted_directions=predicted_directions,
        epsilon=epsilon,
        r=r,
        h=h,
    )

    # Generate one noisy observation.
    X = sample_observation(
        key=key_obs,
        theta=theta,
        n=n,
    )

    # Apply the selected test.
    test_result = test_fn(
        key=key_test,
        X=X,
        predicted_directions=predicted_directions,
        n=n,
        alpha=alpha,
    )

    # Store the rejection indicator as 0/1.
    return test_result.astype(int)


def run_tests_for_configuration(
    key,
    d,
    r,
    n_samples,
    m_predictions,
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

    # One independent PRNG key per Monte Carlo trial.
    keys = jax.random.split(key, n_samples)

    # Vectorize the single-trial simulation and JIT compile it.
    batched_fn = jax.jit(
        jax.vmap(
            lambda kk: run_single_test(
                key=kk,
                d=d,
                m_predictions=m_predictions,
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

    # Ambient dimension d.
    d_values = [100]

    # Numbers of orthogonal prediction directions m to compare.
    m_predictions_values = [10, 25, 50]

    # Number of prediction-error grid points in each panel.
    n_r_values = 30

    # At this value,
    #
    #     A_plus(epsilon, r_max, h) = 0.
    r_max = jnp.sqrt(epsilon**2 + h**2)

    def make_r_values(m_predictions):
        """
        Construct the feasible prediction-error grid for m orthogonal predictions.

        Notes
        -----
        Under orthogonality, ||A_+(epsilon, r)||_2^2 = m A_+^2. Feasibility of a
        mean with norm epsilon therefore requires A_+ <= epsilon / sqrt(m).
        The lower endpoint r_min solves this constraint at equality; r_max makes
        A_+ equal to zero, as in the simulation design in Section 6.
        """
        r_min = jnp.sqrt(
            epsilon**2
            + h**2
            - 2.0 * h * epsilon / jnp.sqrt(m_predictions)
        )

        return jnp.linspace(
            r_min,
            r_max,
            n_r_values,
        )

    # ------------------------------------------------------------------
    # Testing procedures
    # ------------------------------------------------------------------

    tests = {
        "chi_squared_test": chi_squared_test,
        "projection_test": projection_test,
        "bonferroni_projection_test": bonferroni_projection_test,
    }

    # ------------------------------------------------------------------
    # Build the simulation grid
    # ------------------------------------------------------------------

    # Each configuration consists of:
    #
    #   test method,
    #   ambient dimension d,
    #   number of predictions m,
    #   prediction error r.
    #
    # Because the feasible r interval depends on m, make_r_values is
    # evaluated separately for each number of predictions.
    configurations = [
        ((test_name, test_fn), d, m_predictions, r)
        for test_name, test_fn in tests.items()
        for d in d_values
        for m_predictions in m_predictions_values
        for r in make_r_values(m_predictions)
    ]

    # Give every experimental configuration an independent random key.
    outer_keys = jax.random.split(
        key,
        len(configurations),
    )

    # Store trial-level rejection indicators.
    rows = []

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------

    for key_run, config in tqdm(
        zip(outer_keys, configurations),
        total=len(configurations),
        desc="Running experiments",
    ):
        # Unpack this configuration.
        (test_name, test_fn), d, m_predictions, r = config

        # Run n_samples independent Monte Carlo trials.
        test_results = run_tests_for_configuration(
            key=key_run,
            d=d,
            r=r,
            n_samples=n_samples,
            m_predictions=m_predictions,
            epsilon=epsilon,
            h=h,
            n=n,
            alpha=alpha,
            test_fn=test_fn,
        )

        # Transfer results from the JAX device to the host.
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
            ]
        )["test_result"]
        .mean()
        .reset_index()
        .rename(columns={"test_result": "rejection_rate"})
    )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    # Keep display-related modifications separate from the simulation data.
    plot_df = summary_df.copy()

    # Map internal test names to the labels used in the paper figures.
    test_name_map = {
        "chi_squared_test": "Chi-squared",
        "projection_test": "Projection",
        "bonferroni_projection_test": "Adaptive",
    }

    plot_df["test"] = plot_df["test"].replace(test_name_map)

    # Explicit facet order based on increasing number of predictions.
    m_order = [
        f"m = {m} prediction{'s' if m > 1 else ''}"
        for m in sorted(plot_df["m_predictions"].unique())
    ]

    # Create facet labels.
    plot_df["prediction_panel"] = pd.Categorical(
        plot_df["m_predictions"].apply(
            lambda m: f"m = {m} prediction{'s' if m > 1 else ''}"
        ),
        categories=m_order,
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

        # Each panel corresponds to a different number of orthogonal
        # prediction directions.
        #
        # scales="free_x" is used because the feasible error range changes
        # with m_predictions.
        + facet_wrap(
            "~ prediction_panel",
            scales="free_x",
        )
        + theme_bw()
        + labs(
            title="Power across number of orthogonal predictions (d=100)",
            x="Unknown predictions' error (larger is worse)",
            y="Power (higher is better)",
            color="Test",
        )
        + theme(
            # Put the test legend above the panels.
            legend_position="top",

            # Reduce unused whitespace around the legend.
            legend_margin=0,
            legend_box_margin=0,

            # Reduce horizontal spacing between facets.
            panel_spacing_x=0.01,

            # Keep the title close to the plot.
            plot_title=element_text(
                margin={"b": 2}
            ),

            # Reduce spacing after the legend title.
            legend_title=element_text(
                margin={"r": 4}
            ),
        )
    )

    # Save the figure.
    p_r.save(
        "orthogonal_predictions.pdf",
        dpi=300,
        width=8,
        height=4,
    )