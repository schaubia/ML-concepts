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
    ("activation",    "Activation Functions",          "🎯", "ReLU, Sigmoid, Tanh and their properties"),
    ("backprop",      "Backpropagation",                "🔄", "How the error propagates backwards"),
    ("batch_size",    "Batch Size & Gradient Noise",   "🎲", "How batch size affects gradient quality"),
    ("bias_var",      "Bias-Variance Tradeoff",         "↔️", "Decomposing prediction error into bias and variance"),
    ("confusion",     "Confusion Matrix & Metrics",     "🔢", "Precision, recall, F1 and the threshold effect"),
    ("dropout",       "Dropout",                        "💧", "Randomly zeroing neurons to prevent overfitting"),
    ("gradient",      "Gradient & Descent",             "🏔️", "Direction and step size of learning"),
    ("knn",           "K-Nearest Neighbors",            "🔵", "Classify by majority vote of closest points"),
    ("linear_reg",    "Linear Regression",              "📊", "Finding the best-fit line through data"),
    ("logistic_reg",  "Logistic Regression",            "🔀", "Binary classification with sigmoid output"),
    ("loss",          "Loss Function",                  "📉", "How we measure model error"),
    ("lr_schedule",   "Learning Rate Schedulers",       "📅", "Step decay, cosine annealing and warmup"),
    ("neural_net",    "Neural Network Architecture",    "🧬", "Layers, parameters and forward pass"),
    ("neuron",        "Neuron (Perceptron)",             "🔬", "The single computational unit at the core of every network"),
    ("normalization", "Normalization",                   "📐", "Batch norm, layer norm and feature scaling"),
    ("optimizers",    "Optimizers",                     "🚀", "SGD, Momentum, RMSProp and Adam compared"),
    ("overfit",       "Overfitting / Underfitting",     "⚖️", "Too much or too little training"),
    ("regularization","Regularization",                 "🔒", "L1 and L2 penalty to prevent overfitting"),
    ("vanishing_grad","Vanishing & Exploding Gradients","⚡", "Signal death in deep networks and fixes"),
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

# ═══════════════════════════════════════════════════════════════════════════
# REGULARIZATION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "regularization":
    st.title("🔒 Regularization")
    st.markdown("""
    <div class="concept-card">
    Regularization adds a <b>penalty term</b> to the loss to discourage the model from fitting
    noise. It keeps weights small, which generally means a simpler, more generalisable model.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["L1 vs L2 penalty", "Effect on weights"])

    with tab1:
        st.markdown('<div class="formula-box">L2: Loss + λ·Σw²&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;L1: Loss + λ·Σ|w|</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            lam = st.slider("Regularization strength λ", 0.0, 2.0, 0.5, step=0.05)
            reg_type = st.radio("Type", ["L2 (Ridge)", "L1 (Lasso)", "None"])
            st.markdown("---")
            st.markdown("**Key difference:**")
            st.markdown("- **L2** shrinks weights smoothly toward zero")
            st.markdown("- **L1** drives many weights to exactly zero (sparse model)")
        with col2:
            w_vals = np.linspace(-3, 3, 300)
            base_loss = w_vals**2 * 0.3 + 1
            if reg_type == "L2 (Ridge)":
                penalty = lam * w_vals**2
                pen_label = f"L2 penalty (λ={lam})"
            elif reg_type == "L1 (Lasso)":
                penalty = lam * np.abs(w_vals)
                pen_label = f"L1 penalty (λ={lam})"
            else:
                penalty = np.zeros_like(w_vals)
                pen_label = "No penalty"
            total = base_loss + penalty
            opt_w = w_vals[np.argmin(total)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w_vals, y=base_loss, name="Base loss",
                line=dict(color='#AFA9EC', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=w_vals, y=penalty, name=pen_label,
                line=dict(color='#EF9F27', width=2)))
            fig.add_trace(go.Scatter(x=w_vals, y=total, name="Total loss",
                line=dict(color='#E24B4A', width=2.5)))
            fig.add_vline(x=opt_w, line_dash='dot', line_color='#1D9E75',
                annotation_text=f"optimal w = {opt_w:.2f}")
            fig.update_layout(xaxis_title="Weight w", yaxis_title="Loss",
                height=380, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Train a small linear model with different regularization — watch the weights:")
        np.random.seed(42)
        n_reg = 30
        X_reg = np.random.randn(n_reg, 8)
        true_coef = np.array([2.0, -1.5, 0.0, 0.0, 0.8, 0.0, 0.0, -0.3])
        y_reg = X_reg @ true_coef + np.random.randn(n_reg) * 0.5

        lambdas = np.logspace(-3, 1, 60)
        l2_paths, l1_paths = [], []

        for lv in lambdas:
            # Ridge closed form
            I = np.eye(8)
            w_ridge = np.linalg.solve(X_reg.T @ X_reg + lv * I, X_reg.T @ y_reg)
            l2_paths.append(w_ridge)
            # Lasso via coordinate descent (simple)
            w_lasso = np.zeros(8)
            for _ in range(200):
                for j in range(8):
                    r = y_reg - X_reg @ w_lasso + X_reg[:, j] * w_lasso[j]
                    rho = X_reg[:, j] @ r / n_reg
                    w_lasso[j] = np.sign(rho) * max(abs(rho) - lv / 2, 0)
            l1_paths.append(w_lasso.copy())

        l2_paths = np.array(l2_paths)
        l1_paths = np.array(l1_paths)

        reg_choice = st.radio("Regularization type", ["L2 (Ridge)", "L1 (Lasso)"], horizontal=True)
        paths = l2_paths if reg_choice == "L2 (Ridge)" else l1_paths
        colors8 = ['#534AB7','#E24B4A','#1D9E75','#EF9F27','#D4537E','#2196F3','#9C27B0','#FF5722']

        fig2 = go.Figure()
        for j in range(8):
            fig2.add_trace(go.Scatter(x=lambdas, y=paths[:, j],
                name=f"w{j+1} (true={true_coef[j]})",
                line=dict(color=colors8[j], width=2)))
        fig2.update_layout(xaxis_type='log', xaxis_title="λ (log scale)",
            yaxis_title="Weight value", height=360,
            legend=dict(orientation='h', y=1.15, font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("As λ increases, weights shrink. L1 reaches exactly 0 (sparse); L2 approaches 0 smoothly.")


# ═══════════════════════════════════════════════════════════════════════════
# K-NEAREST NEIGHBORS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "knn":
    st.title("🔵 K-Nearest Neighbors (KNN)")
    st.markdown("""
    <div class="concept-card">
    KNN classifies a new point by looking at the <b>K closest training examples</b>
    and taking a majority vote. No training is needed — all the work happens at prediction time.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        k = st.slider("K (number of neighbors)", 1, 15, 3, step=2)
        n_per_class = st.slider("Points per class", 10, 40, 20)
        seed_knn = st.slider("Dataset seed", 0, 20, 7)
        test_x = st.slider("Test point X", -3.0, 3.0, 0.5, step=0.1)
        test_y = st.slider("Test point Y", -3.0, 3.0, 0.5, step=0.1)

    np.random.seed(seed_knn)
    X_a = np.random.randn(n_per_class, 2) + np.array([-1.2, 0.8])
    X_b = np.random.randn(n_per_class, 2) + np.array([1.2, -0.8])
    X_c = np.random.randn(n_per_class, 2) + np.array([0.0, -1.8])
    X_all = np.vstack([X_a, X_b, X_c])
    y_all = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    class_names = ["Class A", "Class B", "Class C"]
    class_colors = ['#534AB7', '#E24B4A', '#1D9E75']

    test_pt = np.array([test_x, test_y])
    dists = np.linalg.norm(X_all - test_pt, axis=1)
    neighbor_idx = np.argsort(dists)[:k]
    neighbor_labels = y_all[neighbor_idx]
    votes = np.bincount(neighbor_labels, minlength=3)
    prediction = np.argmax(votes)

    with col2:
        fig = go.Figure()
        for cls, cname, col in zip([0,1,2], class_names, class_colors):
            mask = y_all == cls
            fig.add_trace(go.Scatter(x=X_all[mask,0], y=X_all[mask,1],
                mode='markers', name=cname,
                marker=dict(color=col, size=8, opacity=0.7)))

        # draw radius circle
        radius = dists[neighbor_idx[-1]] * 1.02
        theta = np.linspace(0, 2*np.pi, 120)
        fig.add_trace(go.Scatter(x=test_x + radius*np.cos(theta),
            y=test_y + radius*np.sin(theta),
            mode='lines', line=dict(color='#EF9F27', width=1.5, dash='dot'),
            name='KNN radius', showlegend=True))

        # highlight neighbors
        for ni in neighbor_idx:
            fig.add_shape(type='line', x0=test_x, y0=test_y,
                x1=X_all[ni,0], y1=X_all[ni,1],
                line=dict(color='#EF9F27', width=1, dash='dot'))

        # test point
        fig.add_trace(go.Scatter(x=[test_x], y=[test_y], mode='markers',
            name=f'Test point → {class_names[prediction]}',
            marker=dict(color=class_colors[prediction], size=16,
                symbol='star', line=dict(color='black', width=1.5))))

        fig.update_layout(xaxis_title="X", yaxis_title="Y", height=420,
            legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for i, (cname, col) in enumerate(zip(class_names, class_colors)):
        c = [c1, c2, c3][i]
        c.metric(f"{cname} votes", votes[i])

    pred_col = class_colors[prediction]
    st.markdown(f"**Prediction: {class_names[prediction]}** ({votes[prediction]}/{k} votes)")

    st.markdown("### Decision boundaries — how K changes the border")
    st.markdown("Low K → jagged boundaries (high variance). High K → smooth boundaries (high bias).")

    from sklearn.neighbors import KNeighborsClassifier
    xx, yy = np.meshgrid(np.linspace(-4,4,120), np.linspace(-4,4,120))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig2 = make_subplots(rows=1, cols=3, subplot_titles=["K=1", "K=5", "K=15"])
    for col_idx, kv in enumerate([1, 5, 15]):
        clf = KNeighborsClassifier(n_neighbors=kv)
        clf.fit(X_all, y_all)
        Z = clf.predict(grid).reshape(xx.shape)
        for cls, c in enumerate(class_colors):
            mask_g = Z == cls
            fig2.add_trace(go.Scatter(
                x=xx.ravel()[mask_g.ravel()], y=yy.ravel()[mask_g.ravel()],
                mode='markers', marker=dict(color=c, size=3, opacity=0.15),
                showlegend=False), row=1, col=col_idx+1)
        for cls, c in enumerate(class_colors):
            mask = y_all == cls
            fig2.add_trace(go.Scatter(x=X_all[mask,0], y=X_all[mask,1],
                mode='markers', marker=dict(color=c, size=6),
                showlegend=False), row=1, col=col_idx+1)
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX & METRICS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "confusion":
    st.title("🔢 Confusion Matrix & Metrics")
    st.markdown("""
    <div class="concept-card">
    A confusion matrix breaks down predictions into <b>TP, FP, FN, TN</b>.
    From these we derive precision, recall, F1 and accuracy — each tells a different story
    about where the model succeeds and fails.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        threshold = st.slider("Classification threshold", 0.01, 0.99, 0.5, step=0.01)
        n_conf = st.slider("Dataset size", 50, 300, 150)
        imbalance = st.slider("Class imbalance (% positive)", 5, 50, 30)
        seed_c = st.slider("Seed", 0, 20, 3)

    np.random.seed(seed_c)
    n_pos = int(n_conf * imbalance / 100)
    n_neg = n_conf - n_pos
    scores_pos = np.clip(np.random.beta(5, 2, n_pos), 0, 1)
    scores_neg = np.clip(np.random.beta(2, 5, n_neg), 0, 1)
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.array([1]*n_pos + [0]*n_neg)
    preds = (scores >= threshold).astype(int)

    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (TP + TN) / n_conf

    with col2:
        cm = np.array([[TP, FP], [FN, TN]])
        labels_cm = [["TP", "FP"], ["FN", "TN"]]
        text = [[f"<b>{labels_cm[i][j]}</b><br>{cm[i,j]}" for j in range(2)] for i in range(2)]

        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Positive","Predicted Negative"],
            y=["Actual Positive","Actual Negative"],
            colorscale=[[0,'#f0effe'],[0.5,'#AFA9EC'],[1,'#534AB7']],
            text=text, texttemplate="%{text}", textfont=dict(size=18),
            showscale=False))
        fig.update_layout(height=320, xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}", help="TP / (TP+FP) — of all positive predictions, how many were correct?")
    c2.metric("Recall", f"{recall:.3f}", help="TP / (TP+FN) — of all actual positives, how many did we catch?")
    c3.metric("F1 Score", f"{f1:.3f}", help="Harmonic mean of precision and recall")
    c4.metric("Accuracy", f"{accuracy:.3f}", help="(TP+TN) / total")

    st.markdown("### Precision–Recall tradeoff across thresholds")
    thresholds = np.linspace(0.01, 0.99, 200)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        p_ = (scores >= t).astype(int)
        tp_ = np.sum((p_==1) & (labels==1))
        fp_ = np.sum((p_==1) & (labels==0))
        fn_ = np.sum((p_==0) & (labels==1))
        pr = tp_ / (tp_ + fp_ + 1e-9)
        rc = tp_ / (tp_ + fn_ + 1e-9)
        precs.append(pr); recs.append(rc)
        f1s.append(2*pr*rc/(pr+rc+1e-9))

    fig2 = make_subplots(rows=1, cols=2,
        subplot_titles=["Precision & Recall vs Threshold", "Precision-Recall curve"])
    fig2.add_trace(go.Scatter(x=thresholds, y=precs, name='Precision',
        line=dict(color='#534AB7', width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=thresholds, y=recs, name='Recall',
        line=dict(color='#E24B4A', width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=thresholds, y=f1s, name='F1',
        line=dict(color='#1D9E75', width=2)), row=1, col=1)
    fig2.add_vline(x=threshold, line_dash='dash', line_color='#EF9F27',
        annotation_text=f"t={threshold:.2f}", row=1, col=1)
    fig2.add_trace(go.Scatter(x=recs, y=precs, mode='lines',
        name='PR curve', line=dict(color='#534AB7', width=2.5),
        showlegend=False), row=1, col=2)
    # mark current threshold on PR curve
    cur_idx = np.argmin(np.abs(thresholds - threshold))
    fig2.add_trace(go.Scatter(x=[recs[cur_idx]], y=[precs[cur_idx]],
        mode='markers', marker=dict(color='#EF9F27', size=12, symbol='circle'),
        name='Current threshold', showlegend=False), row=1, col=2)
    fig2.update_xaxes(title_text="Threshold", row=1, col=1)
    fig2.update_xaxes(title_text="Recall", row=1, col=2)
    fig2.update_yaxes(title_text="Score", row=1, col=1)
    fig2.update_yaxes(title_text="Precision", row=1, col=2)
    fig2.update_layout(height=340, legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# BIAS-VARIANCE TRADEOFF
# ═══════════════════════════════════════════════════════════════════════════
elif section == "bias_var":
    st.title("🎯 Bias-Variance Tradeoff")
    st.markdown("""
    <div class="concept-card">
    Total prediction error = <b>Bias²</b> + <b>Variance</b> + <b>Irreducible noise</b>.<br>
    A model that is too simple has high bias (underfits). A model that is too complex has
    high variance (overfits). The goal is to find the sweet spot.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">MSE = Bias² + Variance + σ²</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        complexity = st.slider("Model complexity", 1, 15, 4,
            help="Higher = more flexible model (higher degree polynomial)")
        noise_bv = st.slider("Irreducible noise σ", 0.1, 2.0, 0.8, step=0.1)
        n_datasets = st.slider("Number of datasets (for variance)", 5, 20, 10)
        seed_bv = st.slider("Seed", 0, 10, 0)

    np.random.seed(seed_bv)
    x_bv = np.linspace(-3, 3, 300)
    true_fn = np.sin(x_bv)

    x_eval = np.linspace(-3, 3, 80)
    all_preds = []
    for i in range(n_datasets):
        x_tr_bv = np.random.uniform(-3, 3, 25)
        y_tr_bv = np.sin(x_tr_bv) + np.random.randn(25) * noise_bv
        try:
            coef = np.polyfit(x_tr_bv, y_tr_bv, complexity)
            pred = np.polyval(coef, x_eval)
            all_preds.append(np.clip(pred, -10, 10))
        except Exception:
            pass
    all_preds = np.array(all_preds)
    mean_pred = all_preds.mean(axis=0)
    variance_curve = all_preds.var(axis=0)
    true_at_eval = np.sin(x_eval)
    bias_sq_curve = (mean_pred - true_at_eval)**2

    avg_bias_sq = float(bias_sq_curve.mean())
    avg_variance = float(variance_curve.mean())
    avg_noise    = float(noise_bv**2)
    total_err    = avg_bias_sq + avg_variance + avg_noise

    with col2:
        fig = go.Figure()
        for i, pred in enumerate(all_preds):
            fig.add_trace(go.Scatter(x=x_eval, y=pred, mode='lines',
                line=dict(color='rgba(83,74,183,0.18)', width=1),
                showlegend=(i == 0), name='Individual model'))
        fig.add_trace(go.Scatter(x=x_eval, y=mean_pred, name='Mean prediction',
            line=dict(color='#534AB7', width=2.5)))
        fig.add_trace(go.Scatter(x=x_eval, y=true_at_eval, name='True function',
            line=dict(color='#1D9E75', width=2, dash='dash')))
        fig.update_layout(xaxis_title="x", yaxis_title="y",
            yaxis_range=[-4, 4], height=360,
            legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bias²", f"{avg_bias_sq:.3f}")
    c2.metric("Variance", f"{avg_variance:.3f}")
    c3.metric("Noise σ²", f"{avg_noise:.3f}")
    c4.metric("Total Error", f"{total_err:.3f}")

    st.markdown("### How each component changes with model complexity")
    complexities = list(range(1, 16))
    bias_vals, var_vals = [], []
    for c_val in complexities:
        preds_c = []
        for i in range(n_datasets):
            np.random.seed(i + seed_bv * 100)
            x_tr_c = np.random.uniform(-3, 3, 25)
            y_tr_c = np.sin(x_tr_c) + np.random.randn(25) * noise_bv
            try:
                coef = np.polyfit(x_tr_c, y_tr_c, c_val)
                preds_c.append(np.clip(np.polyval(coef, x_eval), -15, 15))
            except Exception:
                pass
        if preds_c:
            ap = np.array(preds_c)
            bias_vals.append(float(((ap.mean(axis=0) - true_at_eval)**2).mean()))
            var_vals.append(float(ap.var(axis=0).mean()))
        else:
            bias_vals.append(None); var_vals.append(None)

    noise_line = [noise_bv**2] * len(complexities)
    total_line = [b + v + noise_bv**2 if b is not None else None
                  for b, v in zip(bias_vals, var_vals)]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=complexities, y=bias_vals, name='Bias²',
        line=dict(color='#534AB7', width=2.5), mode='lines+markers'))
    fig2.add_trace(go.Scatter(x=complexities, y=var_vals, name='Variance',
        line=dict(color='#E24B4A', width=2.5), mode='lines+markers'))
    fig2.add_trace(go.Scatter(x=complexities, y=noise_line, name='Noise σ²',
        line=dict(color='#888780', width=1.5, dash='dash')))
    fig2.add_trace(go.Scatter(x=complexities, y=total_line, name='Total Error',
        line=dict(color='#1D9E75', width=2.5), mode='lines+markers'))
    fig2.add_vline(x=complexity, line_dash='dot', line_color='#EF9F27',
        annotation_text=f"complexity={complexity}")
    fig2.update_layout(xaxis_title="Model complexity", yaxis_title="Error",
        yaxis_range=[0, min(max(bias_vals + var_vals + [2]), 8)],
        height=320, legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULERS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "lr_schedule":
    st.title("📅 Learning Rate Schedulers")
    st.markdown("""
    <div class="concept-card">
    A fixed learning rate is rarely optimal throughout training. <b>Schedulers</b> reduce
    the learning rate over time so the model takes large steps early (explore) and
    small steps later (refine).
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        lr0 = st.slider("Initial learning rate", 0.01, 1.0, 0.1, step=0.01)
        total_epochs = st.slider("Total epochs", 20, 200, 100)
        warmup_epochs = st.slider("Warmup epochs", 0, 20, 5)
        step_size = st.slider("Step decay — drop every N epochs", 5, 40, 20)
        step_gamma = st.slider("Step decay — drop factor", 0.1, 0.9, 0.5, step=0.05)

        schedules_sel = st.multiselect("Show schedules",
            ["Constant", "Step Decay", "Exponential Decay", "Cosine Annealing", "Warmup + Cosine"],
            default=["Step Decay", "Cosine Annealing", "Warmup + Cosine"])

    epochs = np.arange(total_epochs)

    def step_decay(e):
        return lr0 * (step_gamma ** (e // step_size))

    def exp_decay(e):
        return lr0 * np.exp(-3 * e / total_epochs)

    def cosine(e):
        return lr0 * 0.5 * (1 + np.cos(np.pi * e / total_epochs))

    def warmup_cosine(e):
        if e < warmup_epochs:
            return lr0 * (e + 1) / max(warmup_epochs, 1)
        t = (e - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return lr0 * 0.5 * (1 + np.cos(np.pi * t))

    schedule_fns = {
        "Constant":           lambda e: lr0,
        "Step Decay":         step_decay,
        "Exponential Decay":  exp_decay,
        "Cosine Annealing":   cosine,
        "Warmup + Cosine":    warmup_cosine,
    }
    sched_colors = {
        "Constant":          '#888780',
        "Step Decay":        '#534AB7',
        "Exponential Decay": '#D85A30',
        "Cosine Annealing":  '#1D9E75',
        "Warmup + Cosine":   '#E24B4A',
    }

    with col2:
        fig = go.Figure()
        for name in schedules_sel:
            fn = schedule_fns[name]
            lr_vals = [fn(e) for e in epochs]
            fig.add_trace(go.Scatter(x=epochs, y=lr_vals, name=name,
                line=dict(color=sched_colors[name], width=2.5)))
        if warmup_epochs > 0 and "Warmup + Cosine" in schedules_sel:
            fig.add_vline(x=warmup_epochs, line_dash='dot', line_color='#EF9F27',
                annotation_text="warmup end")
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Learning rate",
            height=380, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Simulated training loss with each schedule")
    np.random.seed(42)
    base_noise = np.random.randn(total_epochs) * 0.03

    fig2 = go.Figure()
    for name in schedules_sel:
        fn = schedule_fns[name]
        loss_sim = []
        l = 1.0
        for e in range(total_epochs):
            lr_e = fn(e)
            l = l * (1 - lr_e * 0.15) + abs(base_noise[e]) * lr_e
            loss_sim.append(max(l, 0.01))
        fig2.add_trace(go.Scatter(x=epochs, y=loss_sim, name=name,
            line=dict(color=sched_colors[name], width=2)))
    fig2.update_layout(xaxis_title="Epoch", yaxis_title="Simulated loss",
        height=300, legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Simulated — illustrates the qualitative effect of each schedule, not a real training run.")


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "normalization":
    st.title("📐 Normalization")
    st.markdown("""
    <div class="concept-card">
    Normalization rescales inputs or activations to keep values in a useful range.
    This stabilises training, speeds up convergence and reduces sensitivity to weight initialisation.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Feature Scaling", "Batch / Layer Norm"])

    with tab1:
        st.markdown("### Effect of feature scaling on gradient descent")
        col1, col2 = st.columns([1, 2])
        with col1:
            scale_ratio = st.slider("Feature scale ratio (X₂ / X₁)", 1, 50, 20,
                help="Large ratio = very different feature scales")
            scaling = st.radio("Normalization method",
                ["None", "Min-Max [0,1]", "Standardization (z-score)"])
            lr_norm = st.select_slider("Learning rate", [0.001,0.005,0.01,0.05,0.1], value=0.01)
            steps_norm = st.slider("GD steps", 10, 80, 40)

        np.random.seed(42)
        n_norm = 60
        X1 = np.random.randn(n_norm)
        X2 = np.random.randn(n_norm) * scale_ratio
        y_norm = 2*X1 + 0.5*X2 + np.random.randn(n_norm)*0.3

        if scaling == "Min-Max [0,1]":
            X1s = (X1 - X1.min()) / (X1.max() - X1.min() + 1e-9)
            X2s = (X2 - X2.min()) / (X2.max() - X2.min() + 1e-9)
        elif scaling == "Standardization (z-score)":
            X1s = (X1 - X1.mean()) / (X1.std() + 1e-9)
            X2s = (X2 - X2.mean()) / (X2.std() + 1e-9)
        else:
            X1s, X2s = X1, X2

        X_n = np.column_stack([X1s, X2s])

        # gradient descent trajectory
        w = np.array([0.0, 0.0])
        traj = [w.copy()]
        losses_n = []
        for _ in range(steps_norm):
            pred = X_n @ w
            err = pred - y_norm
            loss_n = np.mean(err**2)
            losses_n.append(loss_n)
            grad = 2 * X_n.T @ err / n_norm
            w = w - lr_norm * grad
            traj.append(w.copy())
        traj = np.array(traj)

        # loss surface
        w1r = np.linspace(-3, 5, 60)
        w2r = np.linspace(-3, 5, 60)
        WW2, BB2 = np.meshgrid(w1r, w2r)
        ZZ2 = np.array([[np.mean((X_n @ np.array([ww, bb]) - y_norm)**2)
                         for ww in w1r] for bb in w2r])

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Contour(z=ZZ2, x=w1r, y=w2r,
                colorscale='RdPu', opacity=0.7,
                contours=dict(showlabels=False),
                colorbar=dict(title='MSE', thickness=10)))
            fig.add_trace(go.Scatter(x=traj[:,0], y=traj[:,1],
                mode='lines+markers', name='GD path',
                line=dict(color='#EF9F27', width=2),
                marker=dict(size=5, color='#EF9F27')))
            fig.add_trace(go.Scatter(x=[traj[0,0]], y=[traj[0,1]],
                mode='markers', marker=dict(color='#E24B4A', size=12, symbol='circle'),
                name='Start'))
            fig.add_trace(go.Scatter(x=[traj[-1,0]], y=[traj[-1,1]],
                mode='markers', marker=dict(color='#1D9E75', size=12, symbol='star'),
                name='End'))
            fig.update_layout(xaxis_title='w₁', yaxis_title='w₂',
                height=380, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Without normalization:** elongated elliptical contours → slow, zigzagging descent.  \n"
                    "**With normalization:** rounder contours → faster, straighter path to minimum.")

    with tab2:
        st.markdown("### Batch Norm vs Layer Norm — where they normalise")
        st.markdown("""
        | | Batch Norm | Layer Norm |
        |---|---|---|
        | Normalises over | the **batch** dimension | the **feature** dimension |
        | Typical use | CNNs, MLPs | Transformers, RNNs |
        | Problem with small batches | Yes — statistics are noisy | No |
        | Works at inference without batch | Needs running stats | Yes |
        """)

        col1, col2 = st.columns([1, 2])
        with col1:
            batch_size = st.slider("Batch size", 2, 32, 8)
            n_features = st.slider("Number of features", 2, 16, 6)
            inp_mean = st.slider("Input mean", -3.0, 3.0, 1.5, step=0.5)
            inp_std = st.slider("Input std", 0.1, 5.0, 2.0, step=0.1)

        np.random.seed(42)
        X_bn = np.random.randn(batch_size, n_features) * inp_std + inp_mean

        bn_out = (X_bn - X_bn.mean(axis=0, keepdims=True)) / (X_bn.std(axis=0, keepdims=True) + 1e-5)
        ln_out = (X_bn - X_bn.mean(axis=1, keepdims=True)) / (X_bn.std(axis=1, keepdims=True) + 1e-5)

        with col2:
            import pandas as pd
            fig_bn = make_subplots(rows=1, cols=3,
                subplot_titles=["Raw input", "After Batch Norm", "After Layer Norm"])
            for r, (data, label) in enumerate([(X_bn, "Raw"), (bn_out, "BatchNorm"), (ln_out, "LayerNorm")]):
                for f in range(min(n_features, 8)):
                    fig_bn.add_trace(go.Box(y=data[:,f], name=f"F{f+1}",
                        showlegend=False, marker_color='#534AB7',
                        opacity=0.7), row=1, col=r+1)
            fig_bn.update_layout(height=340)
            st.plotly_chart(fig_bn, use_container_width=True)
            st.caption("Each box = one feature column. Batch Norm centres each feature; Layer Norm centres each sample.")

# ═══════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "logistic_reg":
    st.title("🔀 Logistic Regression")
    st.markdown("""
    <div class="concept-card">
    Logistic regression predicts a <b>probability</b> between 0 and 1 by passing a linear
    combination of inputs through the <b>sigmoid function</b>. It is the simplest classifier
    and the building block of neural network output layers.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Sigmoid & decision boundary", "Training with gradient descent", "Multi-feature decision boundary"])

    with tab1:
        st.markdown('<div class="formula-box">p = σ(w·x + b) = 1 / (1 + e^(−(w·x+b)))</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            w_lg = st.slider("Weight w", -5.0, 5.0, 2.0, step=0.1, key="lg_w")
            b_lg = st.slider("Bias b", -5.0, 5.0, 0.0, step=0.1, key="lg_b")
            threshold_lg = st.slider("Decision threshold", 0.1, 0.9, 0.5, step=0.05)
            decision_boundary = -b_lg / (w_lg + 1e-9)
            st.metric("Decision boundary x*", f"{decision_boundary:.2f}")
            st.info("The boundary is where p = threshold, i.e. w·x + b = 0 (for threshold 0.5).")

        with col2:
            x_lg = np.linspace(-6, 6, 300)
            p_lg = 1 / (1 + np.exp(-(w_lg * x_lg + b_lg)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_lg, y=p_lg, name='p(x)',
                line=dict(color='#534AB7', width=2.5)))
            fig.add_hline(y=threshold_lg, line_dash='dash', line_color='#EF9F27',
                annotation_text=f"threshold={threshold_lg}")
            fig.add_vline(x=decision_boundary, line_dash='dot', line_color='#E24B4A',
                annotation_text=f"x*={decision_boundary:.2f}")
            fig.add_hrect(y0=threshold_lg, y1=1.05, fillcolor='rgba(29,158,117,0.08)', line_width=0)
            fig.add_hrect(y0=-0.05, y1=threshold_lg, fillcolor='rgba(226,75,74,0.08)', line_width=0)
            fig.update_layout(xaxis_title="x", yaxis_title="Predicted probability p",
                yaxis_range=[-0.05, 1.05], height=360,
                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            n_lg = st.slider("Training points", 20, 100, 50, key="lg_n")
            sep = st.slider("Class separation", 0.5, 4.0, 2.0, step=0.1)
            lr_lg = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.3, 0.5], value=0.1, key="lg_lr")
            epochs_lg = st.slider("Epochs", 10, 200, 80, key="lg_ep")

        np.random.seed(42)
        X_lg = np.concatenate([np.random.randn(n_lg//2) - sep/2,
                                np.random.randn(n_lg//2) + sep/2])
        y_lg = np.array([0]*(n_lg//2) + [1]*(n_lg//2), dtype=float)

        def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

        w_t, b_t = 0.0, 0.0
        loss_hist_lg, w_hist, b_hist = [], [], []
        for _ in range(epochs_lg):
            p = sigmoid(w_t * X_lg + b_t)
            loss_v = -np.mean(y_lg * np.log(p + 1e-9) + (1 - y_lg) * np.log(1 - p + 1e-9))
            loss_hist_lg.append(loss_v)
            w_hist.append(w_t); b_hist.append(b_t)
            dw = np.mean((p - y_lg) * X_lg)
            db = np.mean(p - y_lg)
            w_t -= lr_lg * dw
            b_t -= lr_lg * db

        with col2:
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=["Training loss", "Final decision boundary"])
            fig.add_trace(go.Scatter(y=loss_hist_lg, name='BCE Loss',
                line=dict(color='#E24B4A', width=2)), row=1, col=1)

            x_range_lg = np.linspace(X_lg.min()-0.5, X_lg.max()+0.5, 200)
            p_final = sigmoid(w_t * x_range_lg + b_t)
            fig.add_trace(go.Scatter(x=X_lg[y_lg==0], y=np.zeros(n_lg//2),
                mode='markers', name='Class 0',
                marker=dict(color='#534AB7', size=8, symbol='circle'), showlegend=True), row=1, col=2)
            fig.add_trace(go.Scatter(x=X_lg[y_lg==1], y=np.ones(n_lg//2),
                mode='markers', name='Class 1',
                marker=dict(color='#E24B4A', size=8, symbol='circle'), showlegend=True), row=1, col=2)
            fig.add_trace(go.Scatter(x=x_range_lg, y=p_final,
                name='p(x)', line=dict(color='#1D9E75', width=2.5)), row=1, col=2)
            fig.add_hline(y=0.5, line_dash='dash', line_color='#EF9F27', row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="x", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="p(x)", row=1, col=2)
            fig.update_layout(height=360, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Final w", f"{w_t:.3f}")
        c2.metric("Final b", f"{b_t:.3f}")
        final_acc = np.mean(((sigmoid(w_t * X_lg + b_t) >= 0.5).astype(float)) == y_lg)
        c3.metric("Accuracy", f"{final_acc:.1%}")

    with tab3:
        st.markdown("Decision boundary with **two features** — a line in 2D space.")
        col1, col2 = st.columns([1, 2])
        with col1:
            sep2 = st.slider("Class separation (2D)", 0.5, 3.0, 1.5, step=0.1, key="lg2_sep")
            n_2d = st.slider("Points per class", 20, 80, 40, key="lg2_n")
            epochs_2d = st.slider("Training epochs", 50, 500, 200, key="lg2_ep")
            lr_2d = st.select_slider("Learning rate", [0.01,0.05,0.1,0.3], value=0.1, key="lg2_lr")

        np.random.seed(7)
        X0 = np.random.randn(n_2d, 2) - sep2/2
        X1 = np.random.randn(n_2d, 2) + sep2/2
        X2d = np.vstack([X0, X1])
        y2d = np.array([0]*n_2d + [1]*n_2d, dtype=float)

        w2d = np.zeros(2); b2d = 0.0
        for _ in range(epochs_2d):
            z = X2d @ w2d + b2d
            p2 = sigmoid(z)
            dw = X2d.T @ (p2 - y2d) / len(y2d)
            db = np.mean(p2 - y2d)
            w2d -= lr_2d * dw
            b2d -= lr_2d * db

        with col2:
            xx2, yy2 = np.meshgrid(np.linspace(-4,4,100), np.linspace(-4,4,100))
            grid2 = np.c_[xx2.ravel(), yy2.ravel()]
            Z2 = sigmoid(grid2 @ w2d + b2d).reshape(xx2.shape)

            fig = go.Figure()
            fig.add_trace(go.Contour(x=np.linspace(-4,4,100), y=np.linspace(-4,4,100),
                z=Z2, colorscale=[[0,'rgba(83,74,183,0.15)'],[0.5,'white'],[1,'rgba(226,75,74,0.15)']],
                showscale=False, contours=dict(showlabels=False)))
            fig.add_trace(go.Scatter(x=X0[:,0], y=X0[:,1], mode='markers',
                name='Class 0', marker=dict(color='#534AB7', size=8)))
            fig.add_trace(go.Scatter(x=X1[:,0], y=X1[:,1], mode='markers',
                name='Class 1', marker=dict(color='#E24B4A', size=8)))
            if abs(w2d[1]) > 1e-9:
                x_bd = np.linspace(-4, 4, 100)
                y_bd = -(w2d[0]*x_bd + b2d) / w2d[1]
                mask = (y_bd > -4) & (y_bd < 4)
                fig.add_trace(go.Scatter(x=x_bd[mask], y=y_bd[mask],
                    name='Decision boundary', line=dict(color='#1D9E75', width=2.5)))
            fig.update_layout(xaxis_title="Feature 1", yaxis_title="Feature 2",
                xaxis_range=[-4,4], yaxis_range=[-4,4],
                height=400, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════
elif section == "neural_net":
    st.title("🧬 Neural Network Architecture")
    st.markdown("""
    <div class="concept-card">
    A neural network is a stack of <b>layers</b>, each performing a linear transformation
    followed by an activation function. The number of layers and neurons determines
    the model's capacity — and its parameter count.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Architecture explorer", "Forward pass visualised"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Design your network**")
            input_size = st.slider("Input features", 1, 16, 4)
            n_hidden = st.slider("Number of hidden layers", 1, 6, 3)
            hidden_sizes = []
            for i in range(n_hidden):
                h = st.slider(f"Hidden layer {i+1} neurons", 1, 128, [64,32,16,8,4,2][min(i,5)])
                hidden_sizes.append(h)
            output_size = st.slider("Output neurons", 1, 10, 1)
            activation_nn = st.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh"])

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        total_params = sum(layer_sizes[i]*layer_sizes[i+1] + layer_sizes[i+1]
                           for i in range(len(layer_sizes)-1))

        with col2:
            # Draw network diagram as SVG-like plotly figure
            fig = go.Figure()
            max_neurons = max(min(n, 8) for n in layer_sizes)
            n_layers = len(layer_sizes)
            x_positions = np.linspace(0.1, 0.9, n_layers)
            layer_colors = ['#1D9E75'] + ['#534AB7']*n_hidden + ['#E24B4A']

            node_positions = []
            for li, (lx, n_neurons) in enumerate(zip(x_positions, layer_sizes)):
                display_n = min(n_neurons, 8)
                y_positions = np.linspace(0.1, 0.9, display_n)
                node_positions.append((lx, y_positions, n_neurons, display_n))

                # draw connections to next layer
                if li < n_layers - 1:
                    next_lx, next_ypos, _, next_dn = x_positions[li+1], \
                        np.linspace(0.1, 0.9, min(layer_sizes[li+1], 8)), \
                        layer_sizes[li+1], min(layer_sizes[li+1], 8)
                    for yi in y_positions:
                        for yj in np.linspace(0.1, 0.9, min(layer_sizes[li+1], 8)):
                            fig.add_shape(type='line', x0=lx, y0=yi, x1=next_lx, y1=yj,
                                line=dict(color='rgba(150,150,150,0.15)', width=1))

                # draw neurons
                for yi in y_positions:
                    fig.add_shape(type='circle',
                        x0=lx-0.03, y0=yi-0.035, x1=lx+0.03, y1=yi+0.035,
                        fillcolor=layer_colors[li], line=dict(color='white', width=1.5))
                if n_neurons > 8:
                    fig.add_annotation(x=lx, y=0.02, text=f"+{n_neurons-8} more",
                        showarrow=False, font=dict(size=9, color='gray'))

                # layer label
                lname = "Input" if li == 0 else ("Output" if li == n_layers-1 else f"Hidden {li}")
                fig.add_annotation(x=lx, y=1.0, text=f"<b>{lname}</b><br>{n_neurons}",
                    showarrow=False, font=dict(size=11))

            fig.update_layout(xaxis=dict(visible=False, range=[0,1]),
                yaxis=dict(visible=False, range=[-0.05,1.1]),
                height=400, plot_bgcolor='white',
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Parameter count breakdown")
        param_data = []
        for i in range(len(layer_sizes)-1):
            w_count = layer_sizes[i] * layer_sizes[i+1]
            b_count = layer_sizes[i+1]
            param_data.append({
                "Layer": f"Layer {i+1} → {i+2}",
                "Input neurons": layer_sizes[i],
                "Output neurons": layer_sizes[i+1],
                "Weights": w_count,
                "Biases": b_count,
                "Total": w_count + b_count
            })
        import pandas as pd
        df_params = pd.DataFrame(param_data)
        st.dataframe(df_params, use_container_width=True, hide_index=True)
        st.metric("Total trainable parameters", f"{total_params:,}")

    with tab2:
        st.markdown("Watch a single input flow through the network layer by layer.")
        col1, col2 = st.columns([1, 2])
        with col1:
            inp_vals = []
            for i in range(min(input_size, 4)):
                v = st.slider(f"Input x{i+1}", -3.0, 3.0, float(i*0.5 - 0.5), step=0.1, key=f"nn_inp_{i}")
                inp_vals.append(v)
            while len(inp_vals) < input_size:
                inp_vals.append(0.0)
            inp_arr = np.array(inp_vals)

        def act_fn(x, name):
            if name == "ReLU": return np.maximum(0, x)
            if name == "Sigmoid": return 1/(1+np.exp(-np.clip(x,-50,50)))
            return np.tanh(x)

        np.random.seed(99)
        activations = [inp_arr]
        layer_inputs = []
        cur = inp_arr
        for i in range(len(layer_sizes)-1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b_layer = np.zeros(layer_sizes[i+1])
            z = W.T @ cur + b_layer
            layer_inputs.append(z)
            cur = act_fn(z, activation_nn) if i < len(layer_sizes)-2 else z
            activations.append(cur)

        with col2:
            fig = go.Figure()
            for li, (act, lsize) in enumerate(zip(activations, layer_sizes)):
                display = act[:min(len(act), 12)]
                lname = "Input" if li == 0 else ("Output" if li == len(layer_sizes)-1 else f"H{li}")
                fig.add_trace(go.Bar(
                    x=[f"{lname}[{j}]" for j in range(len(display))],
                    y=display,
                    name=lname,
                    marker_color=['#1D9E75' if li==0 else '#E24B4A' if li==len(layer_sizes)-1 else '#534AB7']*len(display),
                    opacity=0.8
                ))
            fig.update_layout(barmode='group', height=380,
                xaxis_title="Neuron", yaxis_title="Activation value",
                legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Weights are random (seed=99). Change inputs to see activations shift.")


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "optimizers":
    st.title("🚀 Optimizers")
    st.markdown("""
    <div class="concept-card">
    Optimizers determine <em>how</em> the gradient is used to update weights.
    Plain SGD can be slow and noisy; modern optimizers like <b>Adam</b> adapt the
    learning rate per-parameter and add momentum to accelerate convergence.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Optimizer | Update rule | Key idea |
    |---|---|---|
    | **SGD** | w ← w − lr·g | Plain gradient step |
    | **Momentum** | v ← β·v + g, w ← w − lr·v | Accumulate velocity |
    | **RMSProp** | s ← β·s + (1-β)·g², w ← w − lr·g/√s | Adapt per-parameter lr |
    | **Adam** | Momentum + RMSProp with bias correction | Best of both worlds |
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        lr_opt = st.select_slider("Learning rate", [0.001,0.005,0.01,0.05,0.1,0.2], value=0.05)
        n_opt_steps = st.slider("Steps", 20, 150, 60)
        beta1 = st.slider("β₁ (Momentum decay)", 0.5, 0.99, 0.9, step=0.01)
        beta2 = st.slider("β₂ (RMSProp decay)", 0.9, 0.999, 0.999, step=0.001)
        surface_type = st.selectbox("Loss surface",
            ["Elongated bowl", "Ravine (ill-conditioned)", "Saddle point"])
        optimizers_sel = st.multiselect("Show optimizers",
            ["SGD", "Momentum", "RMSProp", "Adam"],
            default=["SGD", "Momentum", "Adam"])
        start_w1 = st.slider("Start w₁", -3.0, 3.0, 2.5, step=0.1)
        start_w2 = st.slider("Start w₂", -3.0, 3.0, 2.5, step=0.1)

    if surface_type == "Elongated bowl":
        def loss_surf(w): return w[0]**2 * 0.2 + w[1]**2 * 5.0
        def grad_surf(w): return np.array([0.4*w[0], 10.0*w[1]])
    elif surface_type == "Ravine (ill-conditioned)":
        def loss_surf(w): return (w[0] - w[1])**2 * 10 + (w[0] + w[1])**2 * 0.1
        def grad_surf(w): return np.array([
            20*(w[0]-w[1]) + 0.2*(w[0]+w[1]),
            -20*(w[0]-w[1]) + 0.2*(w[0]+w[1])])
    else:  # saddle
        def loss_surf(w): return w[0]**2 - w[1]**2 + 0.1*(w[0]**3)
        def grad_surf(w): return np.array([2*w[0] + 0.3*w[0]**2, -2*w[1]])

    opt_colors = {"SGD": '#E24B4A', "Momentum": '#534AB7', "RMSProp": '#EF9F27', "Adam": '#1D9E75'}

    def run_optimizer(name, steps):
        w = np.array([start_w1, start_w2], dtype=float)
        path = [w.copy()]
        losses = []
        v = np.zeros(2); s = np.zeros(2); t = 0
        eps = 1e-8
        for _ in range(steps):
            g = grad_surf(w)
            losses.append(loss_surf(w))
            t += 1
            if name == "SGD":
                w = w - lr_opt * g
            elif name == "Momentum":
                v = beta1 * v + g
                w = w - lr_opt * v
            elif name == "RMSProp":
                s = beta2 * s + (1-beta2) * g**2
                w = w - lr_opt * g / (np.sqrt(s) + eps)
            elif name == "Adam":
                v = beta1 * v + (1-beta1) * g
                s = beta2 * s + (1-beta2) * g**2
                v_hat = v / (1 - beta1**t)
                s_hat = s / (1 - beta2**t)
                w = w - lr_opt * v_hat / (np.sqrt(s_hat) + eps)
            w = np.clip(w, -6, 6)
            path.append(w.copy())
        losses.append(loss_surf(w))
        return np.array(path), losses

    trajectories = {name: run_optimizer(name, n_opt_steps) for name in optimizers_sel}

    w1r = np.linspace(-4, 4, 80)
    w2r = np.linspace(-4, 4, 80)
    WW, BB = np.meshgrid(w1r, w2r)
    ZZ = np.array([[loss_surf(np.array([w, b])) for w in w1r] for b in w2r])

    with col2:
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Trajectories on loss surface", "Loss over steps"])

        fig.add_trace(go.Contour(x=w1r, y=w2r, z=ZZ,
            colorscale='RdPu', opacity=0.6, showscale=False,
            contours=dict(showlabels=False)), row=1, col=1)

        for name, (path, losses) in trajectories.items():
            c = opt_colors[name]
            fig.add_trace(go.Scatter(x=path[:,0], y=path[:,1],
                mode='lines+markers', name=name,
                line=dict(color=c, width=2),
                marker=dict(size=4, color=c)), row=1, col=1)
            fig.add_trace(go.Scatter(y=losses, name=name,
                line=dict(color=c, width=2), showlegend=False), row=1, col=2)

        fig.update_xaxes(title_text="w₁", row=1, col=1)
        fig.update_yaxes(title_text="w₂", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_layout(height=420, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Tip:** try the *Ravine* surface — SGD zigzags badly while Adam cuts straight through.")


# ═══════════════════════════════════════════════════════════════════════════
# DROPOUT
# ═══════════════════════════════════════════════════════════════════════════
elif section == "dropout":
    st.title("💧 Dropout")
    st.markdown("""
    <div class="concept-card">
    During training, dropout <b>randomly sets a fraction of neurons to zero</b> at each step.
    This forces the network to learn redundant representations and acts as a strong
    regularizer, significantly reducing overfitting.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["How dropout works", "Effect on overfitting"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            dropout_rate = st.slider("Dropout rate p", 0.0, 0.9, 0.5, step=0.05,
                help="Fraction of neurons set to zero")
            n_neurons_do = st.slider("Neurons in layer", 8, 32, 16)
            seed_do = st.slider("Random seed (= one training step)", 0, 20, 3)
            st.markdown("---")
            st.markdown(f"**Expected active neurons:** {n_neurons_do * (1-dropout_rate):.1f} / {n_neurons_do}")
            st.markdown(f"**Scale factor at inference:** ×{1/(1-dropout_rate+1e-9):.2f}")
            st.info("At **inference** time, dropout is disabled and weights are scaled by (1-p) to compensate.")

        np.random.seed(seed_do)
        activations_do = np.random.randn(n_neurons_do) * 2
        mask = (np.random.rand(n_neurons_do) > dropout_rate).astype(float)
        dropped = activations_do * mask
        # inverted dropout scaling (as used in practice)
        scaled = dropped / (1 - dropout_rate + 1e-9)

        with col2:
            fig = go.Figure()
            colors_do = ['#E24B4A' if m == 0 else '#534AB7' for m in mask]
            fig.add_trace(go.Bar(x=list(range(n_neurons_do)), y=activations_do,
                name='Original', marker_color='#AFA9EC', opacity=0.5))
            fig.add_trace(go.Bar(x=list(range(n_neurons_do)), y=scaled,
                name='After dropout + scale',
                marker_color=colors_do, opacity=0.9))
            fig.update_layout(barmode='overlay', xaxis_title="Neuron index",
                yaxis_title="Activation value", height=360,
                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            active = int(mask.sum())
            st.markdown(f"🔴 **{n_neurons_do - active} neurons dropped** &nbsp;|&nbsp; 🟣 **{active} neurons active**")

    with tab2:
        st.markdown("Simulated train/test loss curves with and without dropout on an overfit-prone network.")
        col1, col2 = st.columns([1, 2])
        with col1:
            do_rate_sim = st.slider("Dropout rate (simulation)", 0.0, 0.8, 0.4, step=0.05, key="do_sim")
            epochs_do = st.slider("Training epochs", 30, 200, 100, key="do_ep")
            capacity = st.slider("Model capacity (higher = more overfit risk)", 1, 5, 3)

        np.random.seed(0)
        ep_range = np.arange(epochs_do)
        overfit_strength = capacity * 0.015

        # Without dropout
        train_no = 0.8 * np.exp(-ep_range * 0.06) + 0.05 + np.random.randn(epochs_do)*0.01
        test_no  = train_no + overfit_strength * ep_range + np.random.randn(epochs_do)*0.015

        # With dropout
        train_do_sim = 0.9 * np.exp(-ep_range * 0.05) + 0.08 + np.random.randn(epochs_do)*0.015
        gap_reduction = (1 - do_rate_sim) * overfit_strength
        test_do_sim  = train_do_sim + gap_reduction * ep_range + np.random.randn(epochs_do)*0.01

        train_no = np.clip(train_no, 0.04, 1)
        test_no  = np.clip(test_no,  0.04, 2)
        train_do_sim = np.clip(train_do_sim, 0.04, 1)
        test_do_sim  = np.clip(test_do_sim,  0.04, 2)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ep_range, y=train_no, name='Train (no dropout)',
                line=dict(color='#534AB7', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=ep_range, y=test_no, name='Test (no dropout)',
                line=dict(color='#E24B4A', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=ep_range, y=train_do_sim, name=f'Train (p={do_rate_sim})',
                line=dict(color='#534AB7', width=2.5)))
            fig.add_trace(go.Scatter(x=ep_range, y=test_do_sim, name=f'Test (p={do_rate_sim})',
                line=dict(color='#1D9E75', width=2.5)))
            fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss",
                height=380, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Dashed = no dropout. Solid = with dropout. The gap between train and test loss is the overfit signal.")


# ═══════════════════════════════════════════════════════════════════════════
# BATCH SIZE & GRADIENT NOISE
# ═══════════════════════════════════════════════════════════════════════════
elif section == "batch_size":
    st.title("🎲 Batch Size & Gradient Noise")
    st.markdown("""
    <div class="concept-card">
    A <b>batch</b> is the subset of training data used to compute one gradient update.
    Large batches give accurate, smooth gradients. Small batches are noisy but can
    generalise better and require less memory.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Batch size | Gradient quality | Memory | Generalisation | Typical use |
    |---|---|---|---|---|
    | 1 (SGD) | Very noisy | Minimal | Often good | Online learning |
    | 8–64 | Moderate noise | Low | Good | Standard DL |
    | 256–2048 | Smooth | High | Can be worse | Large-scale / TPUs |
    | Full dataset | Exact | Huge | Sometimes poor | Classical ML |
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        n_total = st.slider("Total training samples", 100, 500, 200)
        batch_size_sim = st.select_slider("Batch size",
            [1, 4, 8, 16, 32, 64, 128, 200], value=16)
        lr_batch = st.select_slider("Learning rate", [0.001,0.005,0.01,0.05,0.1], value=0.01)
        n_epochs_b = st.slider("Epochs", 5, 50, 20)
        seed_b = st.slider("Seed", 0, 10, 0)

    np.random.seed(seed_b)
    X_b = np.random.randn(n_total, 2)
    true_w_b = np.array([1.5, -0.8])
    y_b = X_b @ true_w_b + np.random.randn(n_total) * 0.5

    def run_batch_gd(batch_sz, lr, epochs):
        w = np.zeros(2)
        loss_per_step = []
        for ep in range(epochs):
            idx = np.random.permutation(n_total)
            X_sh, y_sh = X_b[idx], y_b[idx]
            for start in range(0, n_total, batch_sz):
                Xb = X_sh[start:start+batch_sz]
                yb = y_sh[start:start+batch_sz]
                g = 2 * Xb.T @ (Xb @ w - yb) / len(yb)
                w = w - lr * g
                loss_per_step.append(np.mean((X_b @ w - y_b)**2))
        return w, loss_per_step

    np.random.seed(seed_b)
    w_final, losses_b = run_batch_gd(batch_size_sim, lr_batch, n_epochs_b)

    # compare multiple batch sizes
    batch_sizes_cmp = [1, 16, 64, n_total]
    all_losses = {}
    for bs in batch_sizes_cmp:
        np.random.seed(seed_b)
        _, ls = run_batch_gd(bs, lr_batch, n_epochs_b)
        all_losses[bs] = ls

    with col2:
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Loss trajectory (batch={batch_size_sim})",
                            "Smoothness comparison (same epochs)"])

        fig.add_trace(go.Scatter(y=losses_b, name=f'batch={batch_size_sim}',
            line=dict(color='#534AB7', width=1.5)), row=1, col=1)

        cmp_colors = ['#E24B4A', '#534AB7', '#EF9F27', '#1D9E75']
        max_steps = max(len(v) for v in all_losses.values())
        for (bs, ls), col_c in zip(all_losses.items(), cmp_colors):
            fig.add_trace(go.Scatter(y=ls,
                name=f'B={bs}' if bs < n_total else 'Full batch',
                line=dict(color=col_c, width=1.8)), row=1, col=2)

        fig.update_xaxes(title_text="Gradient step", row=1, col=1)
        fig.update_xaxes(title_text="Gradient step", row=1, col=2)
        fig.update_yaxes(title_text="MSE", row=1, col=1)
        fig.update_yaxes(title_text="MSE", row=1, col=2)
        fig.update_layout(height=380, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Steps per epoch", int(np.ceil(n_total / batch_size_sim)))
    c2.metric("Total gradient steps", len(losses_b))
    c3.metric("Final MSE", f"{losses_b[-1]:.4f}" if losses_b else "—")

    st.markdown("**Noise in small batches can actually help** — it acts like implicit regularization, "
                "helping the optimizer escape sharp minima that generalise poorly.")


# ═══════════════════════════════════════════════════════════════════════════
# VANISHING & EXPLODING GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "vanishing_grad":
    st.title("⚡ Vanishing & Exploding Gradients")
    st.markdown("""
    <div class="concept-card">
    During backpropagation through many layers, gradients are multiplied together.
    If weights are small, the gradient <b>vanishes</b> (→ 0) and early layers stop learning.
    If weights are large, the gradient <b>explodes</b> (→ ∞) and training becomes unstable.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Gradient magnitude through layers", "Fixes: initialisation & residuals"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            n_layers_vg = st.slider("Number of layers", 2, 30, 10)
            weight_scale = st.slider("Average weight magnitude |w|", 0.1, 2.0, 1.0, step=0.05)
            act_vg = st.selectbox("Activation", ["Sigmoid", "Tanh", "ReLU"], key="vg_act")
            st.markdown("---")
            if weight_scale < 0.9:
                st.error("🔴 **Vanishing gradient** — signal dies before reaching early layers")
            elif weight_scale > 1.1:
                st.warning("🟡 **Exploding gradient** — signal grows exponentially")
            else:
                st.success("🟢 **Stable** — gradient magnitude stays reasonable")

        # Simulate gradient magnitude through layers
        np.random.seed(42)
        grad_mags = [1.0]
        act_deriv_avg = {'Sigmoid': 0.25, 'Tanh': 0.42, 'ReLU': 0.5}
        d_act = act_deriv_avg[act_vg]

        layer_grad_mags = []
        g = 1.0
        for l in range(n_layers_vg):
            w_sample = np.random.randn(8, 8) * weight_scale
            g = g * np.mean(np.abs(w_sample)) * d_act
            layer_grad_mags.append(g)

        layers_axis = list(range(1, n_layers_vg + 1))

        with col2:
            fig = go.Figure()
            colors_vg = ['#E24B4A' if g < 1e-3 else '#EF9F27' if g > 100 else '#1D9E75'
                         for g in layer_grad_mags]
            fig.add_trace(go.Bar(x=layers_axis, y=layer_grad_mags,
                marker_color=colors_vg, name='Gradient magnitude'))
            fig.add_hline(y=1e-3, line_dash='dash', line_color='#E24B4A',
                annotation_text="vanishing zone")
            fig.add_hline(y=100, line_dash='dash', line_color='#EF9F27',
                annotation_text="exploding zone")
            fig.update_layout(xaxis_title="Layer (from output → input)",
                yaxis_title="Gradient magnitude",
                yaxis_type='log', height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Also show on linear scale for small networks
        st.markdown("### Gradient magnitude table (first 10 layers)")
        import pandas as pd
        df_vg = pd.DataFrame({
            "Layer (from output)": layers_axis[:10],
            "Gradient magnitude": [f"{g:.2e}" for g in layer_grad_mags[:10]],
            "Status": ["🔴 Vanished" if g < 1e-3 else "🟡 Exploding" if g > 100 else "🟢 OK"
                       for g in layer_grad_mags[:10]]
        })
        st.dataframe(df_vg, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### How to fix vanishing/exploding gradients")
        st.markdown("""
        **1. Weight Initialisation**
        - **Xavier/Glorot** (for Sigmoid, Tanh): initialise weights with std = √(2 / (n_in + n_out))
        - **He initialisation** (for ReLU): initialise weights with std = √(2 / n_in)
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Xavier vs He vs Naive initialisation**")
            n_in_init = st.slider("Layer input size", 4, 512, 64)
            n_out_init = st.slider("Layer output size", 4, 512, 64)

            naive_std  = 1.0
            xavier_std = np.sqrt(2 / (n_in_init + n_out_init))
            he_std     = np.sqrt(2 / n_in_init)

            np.random.seed(42)
            w_naive  = np.random.randn(n_in_init, n_out_init) * naive_std
            w_xavier = np.random.randn(n_in_init, n_out_init) * xavier_std
            w_he     = np.random.randn(n_in_init, n_out_init) * he_std

            fig2 = go.Figure()
            for w_arr, name, col_c in [(w_naive.ravel(), 'Naive (std=1)', '#E24B4A'),
                                       (w_xavier.ravel(), f'Xavier (std={xavier_std:.3f})', '#534AB7'),
                                       (w_he.ravel(), f'He (std={he_std:.3f})', '#1D9E75')]:
                fig2.add_trace(go.Histogram(x=w_arr, name=name, opacity=0.6,
                    marker_color=col_c, nbinsx=40))
            fig2.update_layout(barmode='overlay', xaxis_title="Weight value",
                yaxis_title="Count", height=300, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("**2. Residual connections (skip connections)**")
            st.markdown("""
            ResNets add a shortcut that bypasses one or more layers:

            ```
            output = F(x, W) + x
            ```

            The gradient can now flow directly through the skip connection,
            bypassing the non-linearity entirely — solving vanishing gradients
            in very deep networks (100+ layers).
            """)

            n_res_layers = st.slider("Network depth", 2, 20, 8, key="res_layers")
            with_residual = st.checkbox("Show with residual connections", value=True)

            w_res = 0.85
            g_no_res = [1.0]
            g_with_res = [1.0]
            for l in range(n_res_layers):
                g_no_res.append(g_no_res[-1] * w_res * 0.25)  # sigmoid-like
                g_with_res.append(g_with_res[-1] * (w_res * 0.25 + 1.0) / 2)  # skip helps

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(y=g_no_res, name='Without residual',
                line=dict(color='#E24B4A', width=2.5), mode='lines+markers'))
            if with_residual:
                fig3.add_trace(go.Scatter(y=g_with_res, name='With residual',
                    line=dict(color='#1D9E75', width=2.5), mode='lines+markers'))
            fig3.update_layout(xaxis_title="Layer (from output)",
                yaxis_title="Gradient magnitude",
                height=280, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        **3. Other fixes:**
        - **Gradient clipping** — cap gradient norm at a threshold (common in RNNs)
        - **Batch Normalisation** — keeps activations well-scaled at each layer
        - **ReLU** instead of Sigmoid/Tanh — derivative is 1 for x>0, not squashed
        """)

# ═══════════════════════════════════════════════════════════════════════════
# SINGLE NEURON / PERCEPTRON
# ═══════════════════════════════════════════════════════════════════════════
elif section == "neuron":
    st.title("🔬 Neuron (Perceptron)")
    st.markdown("""
    <div class="concept-card">
    A single neuron is the fundamental building block of every neural network.
    It takes a set of inputs, multiplies each by a learned <b>weight</b>, adds a <b>bias</b>,
    then passes the result through an <b>activation function</b> to produce an output.
    Everything in deep learning is just many of these stacked together.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Live computation diagram",
        "What weights and bias do",
        "One learning step"
    ])

    # ── TAB 1: live computation diagram ──────────────────────────────────
    with tab1:
        st.markdown("Adjust the inputs, weights and bias — watch every value in the equation update live.")

        col_ctrl, col_diag = st.columns([1, 2])
        with col_ctrl:
            st.markdown("**Inputs**")
            x1 = st.slider("x₁", -3.0, 3.0, 1.0, step=0.1, key="n_x1")
            x2 = st.slider("x₂", -3.0, 3.0, 0.5, step=0.1, key="n_x2")
            x3 = st.slider("x₃", -3.0, 3.0, -1.0, step=0.1, key="n_x3")
            st.markdown("**Weights**")
            w1 = st.slider("w₁", -3.0, 3.0, 0.8, step=0.1, key="n_w1")
            w2 = st.slider("w₂", -3.0, 3.0, -0.5, step=0.1, key="n_w2")
            w3 = st.slider("w₃", -3.0, 3.0, 1.2, step=0.1, key="n_w3")
            st.markdown("**Bias & activation**")
            bias = st.slider("bias b", -3.0, 3.0, 0.3, step=0.1, key="n_b")
            act_n = st.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh", "None (linear)"], key="n_act")

        # compute
        z = w1*x1 + w2*x2 + w3*x3 + bias
        act_fns = {
            "ReLU":          lambda v: max(0.0, v),
            "Sigmoid":       lambda v: 1/(1+np.exp(-v)),
            "Tanh":          lambda v: float(np.tanh(v)),
            "None (linear)": lambda v: v,
        }
        output_n = act_fns[act_n](z)

        with col_diag:
            # ── build the diagram with plotly shapes + annotations ──
            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(visible=False, range=[0, 10]),
                yaxis=dict(visible=False, range=[0, 10]),
                height=460, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
            )

            inputs  = [(x1,"x₁",w1,"w₁"), (x2,"x₂",w2,"w₂"), (x3,"x₃",w3,"w₃")]
            ys_in   = [7.5, 5.0, 2.5]
            x_in, x_sum, x_act, x_out = 1.0, 4.2, 6.8, 9.2

            node_style  = dict(fillcolor="#eeedfe", line=dict(color="#534AB7", width=2))
            sum_style   = dict(fillcolor="#fff8e8", line=dict(color="#EF9F27", width=2))
            act_style   = dict(fillcolor="#e8f8f2", line=dict(color="#1D9E75", width=2))
            out_style   = dict(fillcolor="#fdecea", line=dict(color="#E24B4A", width=2.5))

            r = 0.55  # node radius

            # input nodes
            for (xv, xl, wv, wl), yi in zip(inputs, ys_in):
                fig.add_shape(type="circle", x0=x_in-r, y0=yi-r, x1=x_in+r, y1=yi+r, **node_style)
                fig.add_annotation(x=x_in, y=yi, text=f"<b>{xl}</b><br>{xv:.1f}",
                    showarrow=False, font=dict(size=11))

                # connection line to summation node
                contrib = xv * wv
                lw = min(abs(contrib)*1.5 + 0.5, 5)
                lc = "#534AB7" if contrib >= 0 else "#E24B4A"
                fig.add_shape(type="line", x0=x_in+r, y0=yi, x1=x_sum-r, y1=5.0,
                    line=dict(color=lc, width=lw))
                # weight label on the line
                mx, my = (x_in+r + x_sum-r)/2, (yi + 5.0)/2
                fig.add_annotation(x=mx, y=my,
                    text=f"{wl}={wv:.1f}<br>→{contrib:.2f}",
                    showarrow=False, font=dict(size=9, color=lc),
                    bgcolor="rgba(255,255,255,0.75)", borderpad=2)

            # summation node  Σ
            fig.add_shape(type="circle", x0=x_sum-r, y0=5.0-r, x1=x_sum+r, y1=5.0+r, **sum_style)
            fig.add_annotation(x=x_sum, y=5.5, text="<b>Σ</b>", showarrow=False,
                font=dict(size=18, color="#b8860b"))
            fig.add_annotation(x=x_sum, y=4.6,
                text=f"z={z:.3f}", showarrow=False, font=dict(size=10, color="#5a4a00"))

            # bias arrow (comes from below)
            fig.add_shape(type="line", x0=x_sum, y0=5.0-r-1.0, x1=x_sum, y1=5.0-r,
                line=dict(color="#888", width=1.5, dash="dot"))
            fig.add_annotation(x=x_sum, y=5.0-r-1.2,
                text=f"b={bias:.1f}", showarrow=False, font=dict(size=10, color="#555"))

            # line: Σ → activation
            fig.add_shape(type="line", x0=x_sum+r, y0=5.0, x1=x_act-r, y1=5.0,
                line=dict(color="#EF9F27", width=2))
            fig.add_annotation(x=(x_sum+r+x_act-r)/2, y=5.35,
                text=f"z={z:.2f}", showarrow=False, font=dict(size=9, color="#b8860b"))

            # activation node
            short = {"ReLU":"ReLU","Sigmoid":"σ","Tanh":"tanh","None (linear)":"—"}
            fig.add_shape(type="rect", x0=x_act-r*1.3, y0=5.0-r, x1=x_act+r*1.3, y1=5.0+r, **act_style)
            fig.add_annotation(x=x_act, y=5.0,
                text=f"<b>{short[act_n]}</b>", showarrow=False, font=dict(size=14, color="#0f6e56"))

            # line: activation → output
            fig.add_shape(type="line", x0=x_act+r*1.3, y0=5.0, x1=x_out-r, y1=5.0,
                line=dict(color="#E24B4A", width=2.5))

            # output node
            fig.add_shape(type="circle", x0=x_out-r, y0=5.0-r, x1=x_out+r, y1=5.0+r, **out_style)
            fig.add_annotation(x=x_out, y=5.5, text="<b>ŷ</b>", showarrow=False,
                font=dict(size=15, color="#a32d2d"))
            fig.add_annotation(x=x_out, y=4.6,
                text=f"{output_n:.4f}", showarrow=False, font=dict(size=10, color="#a32d2d"))

            st.plotly_chart(fig, use_container_width=True)

        # live equation below the diagram
        st.markdown("**Equation for this neuron right now:**")
        eq_parts = f"w₁·x₁ + w₂·x₂ + w₃·x₃ + b = ({w1:.1f})({x1:.1f}) + ({w2:.1f})({x2:.1f}) + ({w3:.1f})({x3:.1f}) + ({bias:.1f}) = {z:.4f}"
        st.markdown(f'<div class="formula-box">z = {eq_parts}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="formula-box">output = {act_n}(z) = {act_n}({z:.4f}) = <b>{output_n:.4f}</b></div>',
            unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Weighted sum z", f"{z:.4f}")
        m2.metric(f"{act_n}(z)", f"{output_n:.4f}")
        m3.metric("Contributions",
            f"x₁:{w1*x1:.2f}  x₂:{w2*x2:.2f}  x₃:{w3*x3:.2f}  b:{bias:.2f}")

    # ── TAB 2: what weights and bias do ──────────────────────────────────
    with tab2:
        st.markdown("### Weight = slope of sensitivity. Bias = where the neuron 'turns on'.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Effect of weight magnitude and sign (single input)**")
            x_range = np.linspace(-3, 3, 200)
            weights_demo = [2.0, 1.0, 0.3, -1.0, -2.0]
            fig = go.Figure()
            colors_w = ['#534AB7','#1D9E75','#888780','#EF9F27','#E24B4A']
            for wv, cv in zip(weights_demo, colors_w):
                z_line = wv * x_range + 0.0
                out_line = np.array([float(np.tanh(zv)) for zv in z_line])
                fig.add_trace(go.Scatter(x=x_range, y=out_line,
                    name=f"w={wv}", line=dict(color=cv, width=2)))
            fig.update_layout(xaxis_title="Input x", yaxis_title="Output (tanh)",
                title="Changing weight — same bias (b=0)",
                height=300, legend=dict(orientation='h', y=1.15))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Effect of bias — shifts the activation threshold**")
            biases_demo = [-2.0, -1.0, 0.0, 1.0, 2.0]
            fig = go.Figure()
            for bv, cv in zip(biases_demo, colors_w):
                z_line = 1.5 * x_range + bv
                out_line = np.array([1/(1+np.exp(-zv)) for zv in z_line])
                fig.add_trace(go.Scatter(x=x_range, y=out_line,
                    name=f"b={bv}", line=dict(color=cv, width=2)))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                annotation_text="threshold p=0.5")
            fig.update_layout(xaxis_title="Input x", yaxis_title="Output (sigmoid)",
                title="Changing bias — same weight (w=1.5)",
                height=300, legend=dict(orientation='h', y=1.15))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        - A **large positive weight** makes the neuron very sensitive to that input — it fires strongly when x is high.  
        - A **negative weight** makes the neuron *inhibited* by that input — like an "off switch."  
        - The **bias** shifts the curve left or right, controlling the threshold at which the neuron activates, independently of any input.
        """)

    # ── TAB 3: one learning step ─────────────────────────────────────────
    with tab3:
        st.markdown("### Full learning step — forward pass → loss → gradient → weight update")
        st.markdown("A single neuron learning to output a target value from one input.")

        col1, col2 = st.columns([1, 2])
        with col1:
            x_learn  = st.slider("Input x", -3.0, 3.0, 1.5, step=0.1, key="nl_x")
            y_target_n = st.slider("Target output y", -2.0, 2.0, 1.0, step=0.1, key="nl_y")
            w_learn  = st.slider("Current weight w", -3.0, 3.0, -1.0, step=0.1, key="nl_w")
            b_learn  = st.slider("Current bias b",   -3.0, 3.0,  0.0, step=0.1, key="nl_b")
            lr_learn = st.select_slider("Learning rate", [0.01,0.05,0.1,0.3,0.5,1.0], value=0.1, key="nl_lr")
            act_learn = st.selectbox("Activation", ["None (linear)", "ReLU", "Tanh"], key="nl_act2")

        def fwd(w, b, x, act):
            z = w*x + b
            if act == "ReLU":          return max(0.0, z), z
            elif act == "Tanh":        return float(np.tanh(z)), z
            else:                      return z, z

        def act_deriv(z, act):
            if act == "ReLU":    return 1.0 if z > 0 else 0.0
            elif act == "Tanh":  return 1.0 - float(np.tanh(z))**2
            else:                return 1.0

        y_hat, z_learn = fwd(w_learn, b_learn, x_learn, act_learn)
        loss_learn = 0.5*(y_hat - y_target_n)**2
        dl_dyhat   = y_hat - y_target_n
        dyhat_dz   = act_deriv(z_learn, act_learn)
        dz_dw      = x_learn
        dz_db      = 1.0
        grad_w = dl_dyhat * dyhat_dz * dz_dw
        grad_b = dl_dyhat * dyhat_dz * dz_db
        w_new  = w_learn - lr_learn * grad_w
        b_new  = b_learn - lr_learn * grad_b
        y_hat_new, _ = fwd(w_new, b_new, x_learn, act_learn)
        loss_new = 0.5*(y_hat_new - y_target_n)**2

        with col2:
            st.markdown("**Step-by-step breakdown**")
            st.code(f"""
FORWARD PASS
────────────
z    = w·x + b  =  {w_learn:.2f}·{x_learn:.2f} + {b_learn:.2f}  =  {z_learn:.4f}
ŷ    = {act_learn}(z)  =  {y_hat:.4f}
Loss = ½·(ŷ − y)²  =  ½·({y_hat:.4f} − {y_target_n:.2f})²  =  {loss_learn:.4f}

BACKWARD PASS (chain rule)
──────────────────────────
dL/dŷ  = ŷ − y          =  {dl_dyhat:.4f}
dŷ/dz  = {act_learn}'(z)   =  {dyhat_dz:.4f}
dz/dw  = x               =  {dz_dw:.4f}
dz/db  = 1               =  1.0000

grad_w = dL/dŷ · dŷ/dz · dz/dw  =  {grad_w:.4f}
grad_b = dL/dŷ · dŷ/dz · dz/db  =  {grad_b:.4f}

WEIGHT UPDATE  (lr = {lr_learn})
────────────────────────────────
w_new = {w_learn:.4f} − {lr_learn}·({grad_w:.4f})  =  {w_new:.4f}
b_new = {b_learn:.4f} − {lr_learn}·({grad_b:.4f})  =  {b_new:.4f}

RESULT
──────
Loss before: {loss_learn:.4f}
Loss after:  {loss_new:.4f}   {'✅ improved' if loss_new < loss_learn else '⚠️ got worse (lr too large?)'}
""", language="text")

        # show loss landscape and where we moved
        w_range = np.linspace(-3, 3, 200)
        loss_curve = np.array([0.5*(fwd(wv, b_learn, x_learn, act_learn)[0] - y_target_n)**2
                                for wv in w_range])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w_range, y=loss_curve,
            name="Loss vs w", line=dict(color='#AFA9EC', width=2.5)))
        fig.add_scatter(x=[w_learn], y=[loss_learn], mode='markers',
            marker=dict(color='#E24B4A', size=14, symbol='circle'), name='Current w')
        fig.add_annotation(x=w_learn, y=loss_learn, text=f"  before<br>  w={w_learn:.2f}",
            showarrow=True, arrowhead=2, ax=30, ay=-30, font=dict(color='#E24B4A'))
        fig.add_scatter(x=[w_new], y=[loss_new], mode='markers',
            marker=dict(color='#1D9E75', size=14, symbol='star'), name='Updated w')
        fig.add_annotation(x=w_new, y=loss_new, text=f"  after<br>  w={w_new:.2f}",
            showarrow=True, arrowhead=2, ax=30, ay=30, font=dict(color='#1D9E75'))
        fig.add_shape(type='line', x0=w_learn, y0=loss_learn, x1=w_new, y1=loss_new,
            line=dict(color='#EF9F27', width=2, dash='dash'))
        fig.update_layout(xaxis_title="Weight w", yaxis_title="Loss",
            height=340, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        > **This is the full loop:** forward pass computes the output and loss;
        > the chain rule decomposes how much each weight contributed to the error;
        > the update nudges every weight in the direction that reduces loss.
        > Repeat this millions of times across many neurons — that's training a neural network.
        """)
