import streamlit as st

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
    Kernel,
)
import numpy as np
import plotly.graph_objs as go

import SessionState
import acquisition_functions

st.set_page_config(layout="wide")

X_MIN = -5.0
X_MAX = 5.0


def restart_game(session) -> None:
    session.x_sample = np.empty(0)
    session.y_sample = np.empty(0)
    st.experimental_rerun()


def ask_observations(session) -> None:
    col1, col2 = st.beta_columns(2)
    with col1:
        obs_x = np.asarray(
            [st.number_input("x:", min_value=X_MIN, max_value=X_MAX, value=0.0)]
        )
    with col2:
        obs_y = np.asarray([st.number_input("y:", value=1.0)])
    if st.button("Add Observation"):
        session.x_sample = np.concatenate((session.x_sample, obs_x))
        session.y_sample = np.concatenate((session.y_sample, obs_y))


def ask_kernel_fn() -> Kernel:
    kernels = {
        "RBF": RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
        "RationalQuadratic": RationalQuadratic(length_scale=1.0, alpha=0.1),
        "ExpSineSquared": ExpSineSquared(
            length_scale=1.0,
            periodicity=3.0,
            length_scale_bounds=(0.1, 10.0),
            periodicity_bounds=(1.0, 10.0),
        ),
        "DotProduct": ConstantKernel(0.1, (0.01, 10.0))
        * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
        "Matern": Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5),
    }

    kernel_name = st.selectbox("Select kernel:", list(kernels.keys()))
    return kernels[kernel_name]


def ask_acquisition_fn(session, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    acquisition_fn = st.selectbox(
        "Select acquisition function:",
        [
            "Probability of Improvement (PI)",
            "Expected Improvement (EI)",
            "Upper Confidence Bound (UCB)",
        ],
        index=0,
    )

    if acquisition_fn == "Probability of Improvement (PI)":
        epsilon = st.number_input(
            "Choose epsilon (exploitation/exploration):", min_value=0.0, value=0.01
        )
        return acquisition_functions.PIacquisition(
            y_mean,
            y_std,
            session.y_sample.max() if session.y_sample.shape[0] >= 1 else 0,
            epsilon=epsilon,
        )
    elif acquisition_fn == "Expected Improvement (EI)":
        epsilon = st.number_input(
            "Choose epsilon (exploitation/exploration):", min_value=0.0, value=0.01
        )
        return acquisition_functions.EIacquisition(
            y_mean,
            y_std,
            session.y_sample.max() if session.y_sample.shape[0] >= 1 else 0,
            epsilon=epsilon,
        )
    elif acquisition_fn == "Upper Confidence Bound (UCB)":
        beta = st.number_input("Choose beta:", min_value=0.0, value=0.5)
        return y_mean + beta * y_std
    else:
        raise RuntimeError()


session = SessionState.get(x_sample=np.empty(0), y_sample=np.empty(0))

col1, *_, col2 = st.beta_columns([40, 10, 2])
col1.header("Bayesian Optimization (GP)")
if col2.button("↻"):
    restart_game(session)

st.subheader("Observations")
ask_observations(session)

x = np.linspace(X_MIN, X_MAX, 1000)

st.subheader("GP Kernel")
kernel = ask_kernel_fn()

gp = GaussianProcessRegressor(kernel=kernel)
if session.x_sample.shape[0] >= 1:
    gp.fit(session.x_sample[:, None], session.y_sample)
y_mean, y_std = gp.predict(x[:, None], return_std=True)

st.subheader("Acquisition function")
acquisition_y = ask_acquisition_fn(session, y_mean, y_std)

fig = go.Figure(
    [
        go.Scatter(
            x=x,
            y=y_mean,
            line=dict(color="rgba(130, 143, 213, 1)"),
            mode="lines",
            name="surrogate",
        ),
        go.Scatter(
            x=x,  # x, then x reversed
            y=y_mean + y_std,  # upper, then lower reversed
            fillcolor="rgba(141, 202, 212, 0.4)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=x,  # x, then x reversed
            y=y_mean - y_std,  # upper, then lower reversed
            fill="tonexty",
            fillcolor="rgba(141, 202, 212, 0.4)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]
)
fig.add_trace(
    go.Scatter(
        x=session.x_sample,  # x, then x reversed
        y=session.y_sample,  # upper, then lower reversed
        name=f"observations",
        line=dict(color="rgba(182, 77, 58, 1.0)"),
        mode="markers",
    )
)
fig.add_trace(
    go.Scatter(
        x=x,  # x, then x reversed
        y=acquisition_y,  # upper, then lower reversed
        name=f"acquisition",
        fill="tozeroy",
        fillcolor="rgba(219, 179, 210, 0.3)",
        line=dict(color="rgba(219, 179, 210, 0.6)"),
    )
)

fig.update_yaxes(
    range=[-2.1, 2.1],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
)
fig.update_xaxes(
    range=[X_MIN, X_MAX],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
)
fig.update_layout(
    height=750,
)

st.plotly_chart(fig, use_container_width=True)

# bound_mask = np.logical_and(x >= 0, x <= 5)
# next_x = x[bound_mask][acquisition_y[bound_mask].argmax(0)]
next_x = x[acquisition_y.argmax(0)]
st.markdown(f"Next try according to the acquisition function: `x={next_x:.2f}`")

st.markdown("---")
st.markdown("Author: [`Luca Moschella`](https://luca.moschella.dev)")
