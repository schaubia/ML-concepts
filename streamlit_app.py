import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ML Concepts",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.concept-card {
    background: #f8f9fa;
    border-left: 4px solid #534AB7;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}
.formula-box {
    background: #eeedfe;
    border: 1px solid #afa9ec;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-family: monospace;
    font-size: 1.05rem;
    text-align: center;
    margin: 0.5rem 0;
    color: #3C3489;
}
.stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ── Catalogue ────────────────────────────────────────────────────────────────
# (key, name, icon, short description)
CATALOGUE = [
    ("activation", "Activation Functions",      "🎯", "ReLU, Sigmoid, Tanh and their properties"),
    ("backprop",   "Backpropagation",            "🔄", "How the error propagates backwards"),
    ("gradient",   "Gradient & Descent",         "🏔️", "Direction and step size of learning"),
    ("linear_reg", "Linear Regression",          "📊", "Finding the best-fit line through data"),
    ("loss",       "Loss Function",              "📉", "How we measure model error"),
    ("overfit",    "Overfitting / Underfitting", "⚖️", "Too much or too little training"),
]
# alphabetical order for the sidebar index
ALPHA = sorted(CATALOGUE, key=lambda x: x[1].lower())

# ── State ────────────────────────────────────────────────────────────────────
if "section" not in st.session_state:
    st.session_state["section"] = "home"

def go_to(key):
    st.session_state["section"] = key

# ── Sidebar — alphabetical index ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 ML Concepts")
    st.caption("Interactive reference guide")
    st.divider()

    if st.button("🏠  Home", use_container_width=True):
        go_to("home")

    st.markdown("**A–Z Index**")
    for key, name, icon, _ in ALPHA:
        if st.button(f"{icon} {name}", key=f"sb_{key}", use_container_width=True):
            go_to(key)

    st.divider()
    st.caption("Runs fully offline — no internet needed")

section = st.session_state["section"]

# ═══════════════════════════════════════════════════════════════════════════
# HOME — clickable cards
# ═══════════════════════════════════════════════════════════════════════════
if section == "home":
    st.title("Interactive Machine Learning Reference")
    st.markdown("Pick a concept from the cards below or from the A–Z index on the left.")
    st.divider()

    cols = st.columns(3)
    for i, (key, name, icon, desc) in enumerate(CATALOGUE):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="border:0.5px solid #d3d1c7;border-radius:12px;
                        padding:1.2rem;margin-bottom:0.4rem;min-height:100px">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:500;margin:6px 0 4px">{name}</div>
                <div style="font-size:0.85rem;color:#5F5E5A">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open →", key=f"card_{key}", use_container_width=True):
                go_to(key)
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "loss":
    st.title("📉 Loss Function")
    st.markdown("""
    <div class="concept-card">
    A <b>loss function</b> measures how wrong the model is. It takes the prediction and
    the true value and returns a single number — <em>how badly we are doing</em>.
    The goal of training is to minimise this value.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["MSE — Mean Squared Error", "MAE — Mean Absolute Error", "Binary Cross-Entropy"])

    with tab1:
        st.markdown('<div class="formula-box">MSE = (1/n) · Σ (y_true − y_pred)²</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Parameters**")
            n_pts = st.slider("Number of points", 5, 30, 10, key="mse_n")
            noise = st.slider("Data noise", 0.1, 3.0, 1.0, key="mse_noise")
            pred_offset = st.slider("Prediction offset", -3.0, 3.0, 1.0, key="mse_off", step=0.1)
            np.random.seed(42)
            x_data = np.linspace(0, 10, n_pts)
            y_true = 2 * x_data + 1 + np.random.normal(0, noise, n_pts)
            y_pred = 2 * x_data + 1 + pred_offset
            mse_val = np.mean((y_true - y_pred) ** 2)
            mae_val = np.mean(np.abs(y_true - y_pred))
            st.metric("MSE", f"{mse_val:.3f}")
            st.metric("MAE", f"{mae_val:.3f}")
            st.info("MSE penalises large errors more heavily because of the square.")
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers',
                name='True values', marker=dict(color='#534AB7', size=8)))
            fig.add_trace(go.Scatter(x=x_data, y=y_pred, mode='lines',
                name='Prediction', line=dict(color='#E24B4A', width=2)))
            for xi, yt, yp in zip(x_data, y_true, y_pred):
                fig.add_shape(type='line', x0=xi, x1=xi, y0=yt, y1=yp,
                    line=dict(color='#EF9F27', width=1.5, dash='dot'))
            fig.update_layout(title="Data, prediction and errors (orange)",
                xaxis_title="x", yaxis_title="y", height=380,
                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="formula-box">MAE = (1/n) · Σ |y_true − y_pred|</div>', unsafe_allow_html=True)
        st.markdown("MAE vs MSE for different error magnitudes:")
        errors = np.linspace(0, 4, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=errors, y=errors**2, name='MSE contribution (e²)',
            line=dict(color='#534AB7', width=2.5)))
        fig.add_trace(go.Scatter(x=errors, y=errors, name='MAE contribution (|e|)',
            line=dict(color='#1D9E75', width=2.5)))
        fig.update_layout(xaxis_title="Error e", yaxis_title="Contribution to loss",
            height=360, legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        - **MSE** grows quadratically → strongly penalises outliers
        - **MAE** grows linearly → more robust to outliers
        """)

    with tab3:
        st.markdown('<div class="formula-box">BCE = −[y·log(p) + (1−y)·log(1−p)]</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            true_label = st.radio("True class (y)", [1, 0], key="bce_y")
            pred_prob = st.slider("Predicted probability (p)", 0.01, 0.99, 0.7, step=0.01, key="bce_p")
            bce = -(true_label * np.log(pred_prob) + (1 - true_label) * np.log(1 - pred_prob))
            st.metric("BCE Loss", f"{bce:.4f}")
            if bce < 0.3:
                st.success("Very good prediction!")
            elif bce < 1.0:
                st.warning("Moderate error")
            else:
                st.error("Large error!")
        with col2:
            probs = np.linspace(0.01, 0.99, 300)
            loss_y1 = -np.log(probs)
            loss_y0 = -np.log(1 - probs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=probs, y=loss_y1, name='y=1 (want p→1)',
                line=dict(color='#534AB7', width=2.5)))
            fig.add_trace(go.Scatter(x=probs, y=loss_y0, name='y=0 (want p→0)',
                line=dict(color='#D85A30', width=2.5)))
            fig.add_vline(x=pred_prob, line_dash='dash', line_color='#EF9F27',
                annotation_text=f"p={pred_prob:.2f}")
            fig.update_layout(xaxis_title="Predicted probability",
                yaxis_title="Loss", yaxis_range=[0, 5], height=360,
                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# GRADIENT DESCENT
# ═══════════════════════════════════════════════════════════════════════════
elif section == "gradient":
    st.title("🏔️ Gradient & Gradient Descent")
    st.markdown("""
    <div class="concept-card">
    The <b>gradient</b> is a vector that points in the direction of steepest increase of the function.
    To minimise the loss we walk in the <em>opposite</em> direction — that is <b>gradient descent</b>.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["1D simulation", "2D loss surface"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Settings**")
            lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5], value=0.1)
            start_x = st.slider("Starting point x₀", -4.0, 4.0, 3.0, step=0.1)
            n_steps = st.slider("Number of steps", 5, 60, 25)
            loss_type = st.selectbox("Loss shape", ["Parabola x²", "Asymmetric", "With local minimum"])

        def get_loss_and_grad(loss_type):
            if loss_type == "Parabola x²":
                return lambda x: x**2, lambda x: 2*x
            elif loss_type == "Asymmetric":
                return lambda x: 0.5*x**2 + 0.3*x**3 * (x < 0), lambda x: x + 0.9*x**2 * (x < 0)
            else:
                return (lambda x: x**2 + 2*np.sin(2*x),
                        lambda x: 2*x + 4*np.cos(2*x))

        loss_fn, grad_fn = get_loss_and_grad(loss_type)

        xs_hist = [start_x]
        for _ in range(n_steps):
            g = grad_fn(xs_hist[-1])
            xs_hist.append(xs_hist[-1] - lr * g)
            if abs(g) < 1e-5:
                break

        with col2:
            x_range = np.linspace(-5, 5, 400)
            y_range = np.array([loss_fn(xi) for xi in x_range])
            ys_hist = [loss_fn(xi) for xi in xs_hist]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name='Loss curve',
                line=dict(color='#AFA9EC', width=2.5)))
            fig.add_trace(go.Scatter(
                x=xs_hist, y=ys_hist, mode='markers+lines',
                name='Gradient descent steps',
                marker=dict(color=list(range(len(xs_hist))), colorscale='RdYlGn_r',
                    size=9, showscale=True, colorbar=dict(title='Step', thickness=12)),
                line=dict(color='rgba(239,159,39,0.4)', width=1.5, dash='dot')
            ))
            fig.add_scatter(x=[xs_hist[0]], y=[ys_hist[0]], mode='markers',
                marker=dict(color='#E24B4A', size=14, symbol='circle'),
                name='Start', showlegend=True)
            fig.add_scatter(x=[xs_hist[-1]], y=[ys_hist[-1]], mode='markers',
                marker=dict(color='#1D9E75', size=14, symbol='star'),
                name='End', showlegend=True)
            fig.update_layout(xaxis_title="x (parameter)", yaxis_title="Loss",
                height=400, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Starting x", f"{xs_hist[0]:.3f}")
        c2.metric("Final x", f"{xs_hist[-1]:.3f}")
        c3.metric("Steps to converge", len(xs_hist) - 1)

        if lr >= 1.0:
            st.warning("Large learning rate — the model may overshoot the minimum and diverge.")
        elif lr <= 0.05:
            st.info("Small learning rate — convergence is stable but slow.")
        else:
            st.success("Good learning rate — fast and stable descent.")

    with tab2:
        st.markdown("Loss surface with two parameters (w₁, w₂):")
        w1 = np.linspace(-3, 3, 80)
        w2 = np.linspace(-3, 3, 80)
        W1, W2 = np.meshgrid(w1, w2)
        Z = W1**2 + 2*W2**2 + 0.5*np.sin(3*W1)*W2

        fig = go.Figure(data=[go.Surface(z=Z, x=W1, y=W2,
            colorscale='RdPu', opacity=0.85,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)))])
        fig.update_layout(scene=dict(
            xaxis_title='w₁', yaxis_title='w₂', zaxis_title='Loss',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))),
            height=480, margin=dict(l=0, r=0, b=0, t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Drag to rotate. The goal is to find the valley — the minimum loss.")

# ═══════════════════════════════════════════════════════════════════════════
# BACKPROPAGATION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "backprop":
    st.title("🔄 Backpropagation")
    st.markdown("""
    <div class="concept-card">
    <b>Backpropagation</b> is the algorithm used to compute the gradient of every parameter
    in the network. It applies the <em>chain rule</em> from calculus to carry the error
    signal from the output back towards the input.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Manual example — simple network")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Inputs**")
        x_in = st.slider("Input x", 0.1, 3.0, 1.0, step=0.1)
        w1_bp = st.slider("Weight w₁", -3.0, 3.0, 0.5, step=0.1)
        w2_bp = st.slider("Weight w₂", -3.0, 3.0, 2.0, step=0.1)
        y_target = st.slider("Target value y", 0.0, 5.0, 2.0, step=0.1)

        h = x_in * w1_bp
        sigmoid_h = 1 / (1 + np.exp(-h))
        output = sigmoid_h * w2_bp
        loss_bp = 0.5 * (output - y_target) ** 2

        d_loss_out = output - y_target
        d_out_w2 = sigmoid_h
        d_out_sig = w2_bp
        d_sig_h = sigmoid_h * (1 - sigmoid_h)
        d_h_w1 = x_in

        grad_w2 = d_loss_out * d_out_w2
        grad_w1 = d_loss_out * d_out_sig * d_sig_h * d_h_w1

    with col2:
        st.markdown("**Forward pass**")
        st.code(f"""
x        = {x_in:.2f}
h        = x · w1 = {h:.4f}
sigmoid  = {sigmoid_h:.4f}
output   = sigmoid · w2 = {output:.4f}
loss     = 0.5·(output - y)² = {loss_bp:.4f}
""", language="text")
        st.markdown("**Backward pass (gradients)**")
        st.code(f"""
dL/dw2 = {grad_w2:.4f}
dL/dw1 = {grad_w1:.4f}

(via chain rule:)
dL/dw1 = (output-y) · w2 · sigmoid'(h) · x
""", language="text")

    st.markdown("### How weights change after one step")
    lr_bp = st.slider("Learning rate", 0.01, 1.0, 0.1, step=0.01)

    steps = 50
    w1_hist, w2_hist, loss_hist = [w1_bp], [w2_bp], [loss_bp]
    ww1, ww2 = w1_bp, w2_bp
    for _ in range(steps):
        hh = x_in * ww1
        sh = 1 / (1 + np.exp(-hh))
        out = sh * ww2
        l = 0.5 * (out - y_target) ** 2
        dl = out - y_target
        gw2 = dl * sh
        gw1 = dl * ww2 * sh * (1 - sh) * x_in
        ww1 -= lr_bp * gw1
        ww2 -= lr_bp * gw2
        w1_hist.append(ww1)
        w2_hist.append(ww2)
        loss_hist.append(l)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss during training", "Weight evolution"])
    fig.add_trace(go.Scatter(y=loss_hist, name='Loss', line=dict(color='#E24B4A', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=w1_hist, name='w₁', line=dict(color='#534AB7', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=w2_hist, name='w₂', line=dict(color='#1D9E75', width=2)), row=1, col=2)
    fig.update_xaxes(title_text="Step")
    fig.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# OVERFITTING / UNDERFITTING
# ═══════════════════════════════════════════════════════════════════════════
elif section == "overfit":
    st.title("⚖️ Overfitting and Underfitting")
    st.markdown("""
    <div class="concept-card">
    <b>Underfitting</b> — the model is too simple and fails to capture the structure in the data.<br>
    <b>Overfitting</b> — the model has memorised the training data and performs poorly on new examples.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        degree = st.slider("Polynomial degree", 1, 15, 3)
        noise_of = st.slider("Noise", 0.1, 1.5, 0.5, step=0.1)
        n_train = st.slider("Training points", 8, 25, 12)

    np.random.seed(7)
    x_tr = np.sort(np.random.uniform(-3, 3, n_train))
    y_tr = np.sin(x_tr) + np.random.normal(0, noise_of, n_train)
    x_test = np.sort(np.random.uniform(-3, 3, 40))
    y_test = np.sin(x_test) + np.random.normal(0, noise_of, 40)

    coeffs = np.polyfit(x_tr, y_tr, degree)
    p = np.poly1d(coeffs)
    x_line = np.linspace(-3.5, 3.5, 300)
    y_line = p(x_line)
    train_mse = np.mean((p(x_tr) - y_tr) ** 2)
    test_mse = np.mean((p(x_test) - y_test) ** 2)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_tr, y=y_tr, mode='markers',
            name='Training data', marker=dict(color='#534AB7', size=9)))
        fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers',
            name='Test data', marker=dict(color='#1D9E75', size=7, symbol='x')))
        fig.add_trace(go.Scatter(x=x_line, y=np.sin(x_line),
            name='True function', line=dict(color='#888780', dash='dash', width=1.5)))
        clip = np.clip(y_line, -4, 4)
        fig.add_trace(go.Scatter(x=x_line, y=clip,
            name=f'Polynomial (degree {degree})', line=dict(color='#E24B4A', width=2.5)))
        fig.update_layout(xaxis_title="x", yaxis_title="y",
            yaxis_range=[-3, 3], height=380, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train MSE", f"{train_mse:.4f}")
    c2.metric("Test MSE", f"{test_mse:.4f}")
    ratio = test_mse / (train_mse + 1e-9)
    c3.metric("Test / Train ratio", f"{ratio:.2f}")

    if degree <= 2:
        st.warning("**Underfitting** — the model is too simple to capture the sine curve.")
    elif degree >= 9:
        st.error('**Overfitting** — the model bends too much and memorises the noise.')
    else:
        st.success("**Good balance** — the model captures the structure without memorising noise.")

    st.markdown("### How MSE depends on model complexity")
    degrees = list(range(1, 14))
    tr_mses, te_mses = [], []
    for d in degrees:
        try:
            c = np.polyfit(x_tr, y_tr, d)
            pp = np.poly1d(c)
            tr_mses.append(np.mean((pp(x_tr) - y_tr) ** 2))
            te_mses.append(min(np.mean((pp(x_test) - y_test) ** 2), 20))
        except Exception:
            tr_mses.append(None); te_mses.append(None)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=degrees, y=tr_mses, name='Train MSE',
        line=dict(color='#534AB7', width=2.5), mode='lines+markers'))
    fig2.add_trace(go.Scatter(x=degrees, y=te_mses, name='Test MSE',
        line=dict(color='#E24B4A', width=2.5), mode='lines+markers'))
    fig2.add_vline(x=degree, line_dash='dash', line_color='#EF9F27',
        annotation_text=f"Selected degree: {degree}")
    fig2.update_layout(xaxis_title="Polynomial degree",
        yaxis_title="MSE", yaxis_range=[0, 5], height=300,
        legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "activation":
    st.title("🎯 Activation Functions")
    st.markdown("""
    <div class="concept-card">
    Activation functions introduce <b>non-linearity</b> into the neural network.
    Without them, no matter how many layers we stack, the network could only learn linear transformations.
    </div>
    """, unsafe_allow_html=True)

    funcs = {
        "ReLU": (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
        "Sigmoid": (lambda x: 1/(1+np.exp(-np.clip(x,-10,10))),
                    lambda x: (1/(1+np.exp(-np.clip(x,-10,10)))) * (1 - 1/(1+np.exp(-np.clip(x,-10,10))))),
        "Tanh": (lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2),
        "Leaky ReLU": (lambda x: np.where(x>0, x, 0.1*x), lambda x: np.where(x>0, 1, 0.1)),
        "ELU": (lambda x: np.where(x>0, x, np.exp(x)-1),
                lambda x: np.where(x>0, 1, np.exp(x))),
    }

    selected = st.multiselect("Functions to compare", list(funcs.keys()),
        default=["ReLU", "Sigmoid", "Tanh"])

    x_act = np.linspace(-4, 4, 300)
    colors = ['#534AB7', '#E24B4A', '#1D9E75', '#EF9F27', '#D4537E']

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Function f(x)", "Derivative f'(x)"])
    for i, name in enumerate(selected):
        f, df = funcs[name]
        fig.add_trace(go.Scatter(x=x_act, y=f(x_act), name=name,
            line=dict(color=colors[i % len(colors)], width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_act, y=df(x_act), name=f"{name}'",
            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
            showlegend=False), row=1, col=2)

    fig.update_layout(height=380, legend=dict(orientation='h', y=1.12))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="f'(x)", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Properties")
    props = {
        "ReLU":       ("Simple, fast", "Dead neurons for x<0", "Hidden layers"),
        "Sigmoid":    ("Output in [0,1]", "Vanishing gradient", "Output (binary classification)"),
        "Tanh":       ("Zero-centred", "Vanishing gradient", "Hidden layers"),
        "Leaky ReLU": ("No dead neurons", "Extra hyperparameter", "Hidden layers"),
        "ELU":        ("Smooth, faster convergence", "Slower to compute", "Hidden layers"),
    }
    rows = [(n,) + props[n] for n in selected if n in props]
    if rows:
        import pandas as pd
        df_props = pd.DataFrame(rows, columns=["Function", "Advantage", "Disadvantage", "Use case"])
        st.dataframe(df_props, use_container_width=True, hide_index=True)

    st.markdown("### Interactive example — effect of the activation")
    x_demo = st.slider("Input x", -4.0, 4.0, 1.5, step=0.1)
    chosen_fn = st.selectbox("Select function", list(funcs.keys()))
    f_demo, df_demo = funcs[chosen_fn]
    out_demo = f_demo(np.array([x_demo]))[0]
    grad_demo = df_demo(np.array([x_demo]))[0]
    d1, d2 = st.columns(2)
    d1.metric(f"{chosen_fn}({x_demo:.1f})", f"{out_demo:.4f}")
    d2.metric(f"Gradient at x={x_demo:.1f}", f"{grad_demo:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "linear_reg":
    st.title("📊 Linear Regression")
    st.markdown("""
    <div class="concept-card">
    Linear regression finds a line <b>y = w·x + b</b> that minimises the MSE
    between predicted and true values. The weight <code>w</code> and bias <code>b</code>
    are found via gradient descent or the analytical (closed-form) solution.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Interactive regression", "Gradient descent over w and b"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            true_w = st.slider("True w", -3.0, 3.0, 1.5, step=0.1, key="lr_w")
            true_b = st.slider("True b", -3.0, 3.0, 0.5, step=0.1, key="lr_b")
            noise_lr = st.slider("Noise", 0.1, 3.0, 1.0, step=0.1, key="lr_n")
            n_lr = st.slider("Number of points", 10, 60, 25, key="lr_pts")

        np.random.seed(42)
        x_lr = np.random.uniform(-4, 4, n_lr)
        y_lr = true_w * x_lr + true_b + np.random.normal(0, noise_lr, n_lr)

        X_mat = np.column_stack([x_lr, np.ones(n_lr)])
        params = np.linalg.lstsq(X_mat, y_lr, rcond=None)[0]
        w_hat, b_hat = params

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_lr, y=y_lr, mode='markers',
                name='Data', marker=dict(color='#534AB7', size=7)))
            x_fit = np.linspace(-4.5, 4.5, 100)
            fig.add_trace(go.Scatter(x=x_fit, y=w_hat*x_fit + b_hat,
                name=f'Fitted: y={w_hat:.2f}x + {b_hat:.2f}',
                line=dict(color='#E24B4A', width=2.5)))
            fig.add_trace(go.Scatter(x=x_fit, y=true_w*x_fit + true_b,
                name=f'True: y={true_w}x + {true_b}',
                line=dict(color='#1D9E75', width=1.5, dash='dash')))
            fig.update_layout(xaxis_title="x", yaxis_title="y",
                height=360, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fitted w", f"{w_hat:.3f}", f"{w_hat - true_w:+.3f}")
        c2.metric("Fitted b", f"{b_hat:.3f}", f"{b_hat - true_b:+.3f}")
        c3.metric("Train MSE", f"{np.mean((w_hat*x_lr+b_hat - y_lr)**2):.3f}")
        c4.metric("R²", f"{1 - np.var(y_lr - (w_hat*x_lr+b_hat))/np.var(y_lr):.4f}")

    with tab2:
        st.markdown("Loss surface over parameters **w** and **b**:")
        ws = np.linspace(-3, 3, 60)
        bs = np.linspace(-3, 3, 60)
        WW, BB = np.meshgrid(ws, bs)
        ZZ = np.zeros_like(WW)
        for i in range(len(ws)):
            for j in range(len(bs)):
                preds = WW[j, i] * x_lr + BB[j, i]
                ZZ[j, i] = np.mean((preds - y_lr) ** 2)

        fig3 = go.Figure(data=[go.Contour(
            z=ZZ, x=ws, y=bs,
            colorscale='RdPu',
            contours=dict(showlabels=True, labelfont=dict(size=10)),
            colorbar=dict(title='MSE', thickness=12)
        )])
        fig3.add_scatter(x=[w_hat], y=[b_hat], mode='markers',
            marker=dict(color='#1D9E75', size=14, symbol='star'),
            name='Found minimum')
        fig3.update_layout(xaxis_title='w', yaxis_title='b',
            height=400, legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("The green star is the MSE minimum — the optimal w and b.")
