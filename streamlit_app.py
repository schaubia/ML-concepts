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
    # ── Machine Learning ──
    ("bias_var",      "Bias-Variance Tradeoff",         "↔️", "Decomposing prediction error into bias and variance"),
    ("confusion",     "Confusion Matrix & Metrics",     "🔢", "Precision, recall, F1 and the threshold effect"),
    ("decision_tree", "Decision Tree",                  "🌳", "Recursive feature splits that form a tree of rules"),
    ("gradient",      "Gradient & Descent",             "🏔️", "Direction and step size of learning"),
    ("knn",           "K-Nearest Neighbors",            "🔵", "Classify by majority vote of closest points"),
    ("linear_reg",    "Linear Regression",              "📊", "Finding the best-fit line through data"),
    ("logistic_reg",  "Logistic Regression",            "🔀", "Binary classification with sigmoid output"),
    ("loss",          "Loss Function",                  "📉", "How we measure model error"),
    ("naive_bayes",   "Naive Bayes",                    "📊", "Probabilistic classifier using Bayes theorem"),
    ("overfit",       "Overfitting / Underfitting",     "⚖️", "Too much or too little training"),
    ("pca",           "PCA",                            "🔍", "Dimensionality reduction via principal components"),
    ("regularization","Regularization",                 "🔒", "L1 and L2 penalty to prevent overfitting"),
    # ── Deep Learning ──
    ("activation",    "Activation Functions",           "🎯", "ReLU, Sigmoid, Tanh and their properties"),
    ("attention",     "Attention Mechanism",            "👁️", "How transformers focus on relevant input tokens"),
    ("backprop",      "Backpropagation",                "🔄", "How the error propagates backwards"),
    ("batch_size",    "Batch Size & Gradient Noise",    "🎲", "How batch size affects gradient quality"),
    ("central_tendency", "Central Tendency",           "📊", "Mean, median and mode — summarising where data is centred"),
    ("cnn",           "Convolutional Layer (CNN)",      "🖼️", "Kernel sliding over input to detect local patterns"),
    ("dispersion",    "Dispersion",                     "📏", "Variance, std dev, IQR — how spread out data is"),
    ("rnn",           "Recurrent Networks",             "🔁", "RNNs, LSTMs and GRUs — memory across time"),
    ("dropout",       "Dropout",                        "💧", "Randomly zeroing neurons to prevent overfitting"),
    ("lr_schedule",   "Learning Rate Schedulers",       "📅", "Step decay, cosine annealing and warmup"),
    ("neural_net",    "Neural Network Architecture",    "🧬", "Layers, parameters and forward pass"),
    ("neuron",        "Neuron (Perceptron)",             "🔬", "The single computational unit at the core of every network"),
    ("normalization", "Normalization",                   "📐", "Batch norm, layer norm and feature scaling"),
    ("optimizers",    "Optimizers",                     "🚀", "SGD, Momentum, RMSProp and Adam compared"),
    ("vanishing_grad","Vanishing & Exploding Gradients","⚡", "Signal death in deep networks and fixes"),
    # ── Agentic AI ──
    ("agent_memory", "Agent Memory Types",        "🧠", "In-context, external (RAG) and parametric memory"),
    ("multi_agent",  "Multi-Agent Systems",        "🤝", "Orchestrators, workers and message passing"),
    ("planning",     "Planning & Task Decomposition","🗺️","Breaking goals into subtasks and dependency graphs"),
    ("react_loop",   "ReAct Loop",                 "🔄", "Reason → Act → Observe cycle for tool-using agents"),
    ("tool_use",     "Tool Use",                   "🔧", "How agents call functions and parse results"),
    # ── Math Foundations ──
    ("chain_rule",    "Chain Rule",                     "🔗", "Derivative of composite functions — the engine of backprop"),
    ("derivative",    "Derivative",                     "📐", "Instantaneous rate of change and the tangent line"),
    ("dot_product",   "Dot Product",                    "·",  "Multiply two vectors into a scalar — similarity and projection"),
    ("embeddings",    "Embeddings",                     "🗺️", "Word2Vec, GloVe and contextual vectors — meaning as geometry"),
    ("eigenvalues",   "Eigenvalues & Eigenvectors",     "λ",  "Directions a matrix stretches without rotating"),
    ("integral",      "Integral",                       "∫",  "Area under a curve and accumulation"),
    ("matrix_ops",    "Matrix Operations",              "🔲", "Addition, multiplication, determinant, transpose and norms"),
    ("partial_deriv", "Partial Derivatives",            "∂",  "Rate of change along one dimension of a multivariable function"),
    ("probability",   "Probability Distributions",      "🎲", "Normal, Binomial, Poisson and their shapes"),
    ("svd",           "SVD — Matrix Decomposition",     "✂️", "Singular values, low-rank approximation and compression"),
    ("vectors",       "Vectors",                        "➡️", "Direction and magnitude — the language of ML data"),
    ("vector_spaces",  "Vector Spaces",                  "🧭", "Basis, span, projection and orthogonality"),
    ("vector_norms",  "Vector Norms",                   "‖·‖", "L1, L2 and Lp norms — measuring size and distance"),
]

ML_KEYS     = {"bias_var","confusion","decision_tree","gradient","knn","linear_reg","logistic_reg",
               "loss","naive_bayes","overfit","pca","regularization"}
DL_KEYS     = {"activation","attention","backprop","batch_size","cnn","dropout","lr_schedule",
               "neural_net","neuron","normalization","optimizers","rnn","vanishing_grad"}
MATH_KEYS   = {"chain_rule","derivative","dot_product","eigenvalues","embeddings","integral","matrix_ops",
               "partial_deriv","probability","svd","vectors","vector_norms","vector_spaces"}
AGENT_KEYS  = {"react_loop","tool_use","planning","agent_memory","multi_agent"}
STAT_KEYS   = {"central_tendency","dispersion"}
# alphabetical within each group
ALPHA_ML   = sorted([c for c in CATALOGUE if c[0] in ML_KEYS],   key=lambda x: x[1].lower())
ALPHA_DL   = sorted([c for c in CATALOGUE if c[0] in DL_KEYS],   key=lambda x: x[1].lower())
ALPHA_MATH = sorted([c for c in CATALOGUE if c[0] in MATH_KEYS], key=lambda x: x[1].lower())
ALPHA_AGENT = sorted([c for c in CATALOGUE if c[0] in AGENT_KEYS], key=lambda x: x[1].lower())
ALPHA_STAT  = sorted([c for c in CATALOGUE if c[0] in STAT_KEYS],  key=lambda x: x[1].lower())

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

    st.markdown("**Math Foundations**")
    for key, name, icon, _ in ALPHA_MATH:
        if st.button(name, key=f"sb_{key}", use_container_width=True):
            go_to(key)
    st.markdown("**Machine Learning**")
    for key, name, icon, _ in ALPHA_ML:
        if st.button(name, key=f"sb_{key}", use_container_width=True):
            go_to(key)
    st.markdown("**Deep Learning**")
    for key, name, icon, _ in ALPHA_DL:
        if st.button(name, key=f"sb_{key}", use_container_width=True):
            go_to(key)
    st.markdown("**Statistics & Data Science**")
    for key, name, icon, _ in ALPHA_STAT:
        if st.button(name, key=f"sb_{key}", use_container_width=True):
            go_to(key)
    st.markdown("**Agentic AI**")
    for key, name, icon, _ in ALPHA_AGENT:
        if st.button(name, key=f"sb_{key}", use_container_width=True):
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

    def render_cards(items):
        cols = st.columns(3)
        for i, (key, name, icon, desc) in enumerate(items):
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

    st.markdown("### 📐 Math Foundations")
    render_cards(ALPHA_MATH)
    st.markdown("### 🤖 Machine Learning")
    render_cards(ALPHA_ML)
    st.markdown("### 🧠 Deep Learning")
    render_cards(ALPHA_DL)
    st.markdown("### 📊 Statistics & Data Science")
    render_cards(ALPHA_STAT)
    st.markdown("### 🤝 Agentic AI")
    render_cards(ALPHA_AGENT)

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

# ═══════════════════════════════════════════════════════════════════════════
# VECTORS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "vectors":
    st.title("➡️ Vectors")
    st.markdown("""
    <div class="concept-card">
    A <b>vector</b> is an ordered list of numbers representing a point or direction in space.
    In ML every data sample, every weight layer output, and every gradient is a vector.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["2D visualisation", "Dot product & similarity", "Vector operations"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            ax = st.slider("a — x", -4.0, 4.0,  2.0, step=0.1, key="v_ax")
            ay = st.slider("a — y", -4.0, 4.0,  1.0, step=0.1, key="v_ay")
            bx = st.slider("b — x", -4.0, 4.0, -1.0, step=0.1, key="v_bx")
            by = st.slider("b — y", -4.0, 4.0,  3.0, step=0.1, key="v_by")
            show_sum = st.checkbox("Show a + b", value=True)
            show_diff= st.checkbox("Show a − b", value=False)

        mag_a = np.sqrt(ax**2 + ay**2)
        mag_b = np.sqrt(bx**2 + by**2)
        dot   = ax*bx + ay*by
        cos_t = dot / (mag_a * mag_b + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))

        with col2:
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=ax, y1=ay,
                line=dict(color="#534AB7", width=3))
            fig.add_annotation(x=ax, y=ay, text=f"<b>a</b> [{ax:.1f}, {ay:.1f}]",
                showarrow=True, arrowhead=2, arrowcolor="#534AB7",
                font=dict(color="#534AB7", size=12), ax=15, ay=-15)

            fig.add_shape(type="line", x0=0, y0=0, x1=bx, y1=by,
                line=dict(color="#E24B4A", width=3))
            fig.add_annotation(x=bx, y=by, text=f"<b>b</b> [{bx:.1f}, {by:.1f}]",
                showarrow=True, arrowhead=2, arrowcolor="#E24B4A",
                font=dict(color="#E24B4A", size=12), ax=-15, ay=-15)

            if show_sum:
                sx, sy = ax+bx, ay+by
                fig.add_shape(type="line", x0=0, y0=0, x1=sx, y1=sy,
                    line=dict(color="#1D9E75", width=2.5, dash="dash"))
                fig.add_annotation(x=sx, y=sy, text=f"<b>a+b</b> [{sx:.1f},{sy:.1f}]",
                    showarrow=True, arrowhead=2, arrowcolor="#1D9E75",
                    font=dict(color="#1D9E75", size=11), ax=20, ay=10)
                fig.add_shape(type="line", x0=ax, y0=ay, x1=sx, y1=sy,
                    line=dict(color="#E24B4A", width=1, dash="dot"))
                fig.add_shape(type="line", x0=bx, y0=by, x1=sx, y1=sy,
                    line=dict(color="#534AB7", width=1, dash="dot"))

            if show_diff:
                dx2, dy2 = ax-bx, ay-by
                fig.add_shape(type="line", x0=0, y0=0, x1=dx2, y1=dy2,
                    line=dict(color="#EF9F27", width=2.5, dash="dash"))
                fig.add_annotation(x=dx2, y=dy2, text=f"<b>a−b</b>",
                    showarrow=True, arrowhead=2, arrowcolor="#EF9F27",
                    font=dict(color="#EF9F27", size=11), ax=20, ay=10)

            # angle arc
            theta_a = np.arctan2(ay, ax)
            theta_b = np.arctan2(by, bx)
            arc_r = 0.6
            thetas = np.linspace(min(theta_a, theta_b), max(theta_a, theta_b), 40)
            fig.add_trace(go.Scatter(x=arc_r*np.cos(thetas), y=arc_r*np.sin(thetas),
                mode='lines', line=dict(color='gray', width=1, dash='dot'),
                showlegend=False))
            mid_t = (theta_a + theta_b)/2
            fig.add_annotation(x=arc_r*1.4*np.cos(mid_t), y=arc_r*1.4*np.sin(mid_t),
                text=f"{angle:.1f}°", showarrow=False, font=dict(size=11, color='gray'))

            lim = max(abs(ax),abs(ay),abs(bx),abs(by),abs(ax+bx),abs(ay+by)) + 1
            fig.update_layout(xaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor='#ccc'),
                yaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor='#ccc',
                    scaleanchor='x', scaleratio=1),
                height=420, plot_bgcolor='white',
                showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("|a|  magnitude", f"{mag_a:.3f}")
        c2.metric("|b|  magnitude", f"{mag_b:.3f}")
        c3.metric("a·b  dot product", f"{dot:.3f}")
        c4.metric("Angle between", f"{angle:.1f}°")

    with tab2:
        st.markdown('<div class="formula-box">a · b = Σ aᵢbᵢ = |a||b|cos θ</div>', unsafe_allow_html=True)
        st.markdown("The dot product measures **how aligned** two vectors are.")
        col1, col2 = st.columns([1,2])
        with col1:
            angle_deg = st.slider("Angle between vectors (degrees)", 0, 180, 45)
            mag_demo  = st.slider("Magnitude of both vectors", 0.5, 3.0, 1.5, step=0.1)
        angle_rad = np.radians(angle_deg)
        dp = mag_demo**2 * np.cos(angle_rad)
        with col2:
            fig = go.Figure()
            fig.add_shape(type="line", x0=0,y0=0, x1=mag_demo,y1=0,
                line=dict(color="#534AB7",width=3))
            fig.add_annotation(x=mag_demo,y=0.15, text="a", font=dict(color="#534AB7",size=14), showarrow=False)
            fig.add_shape(type="line", x0=0,y0=0,
                x1=mag_demo*np.cos(angle_rad), y1=mag_demo*np.sin(angle_rad),
                line=dict(color="#E24B4A",width=3))
            fig.add_annotation(x=mag_demo*np.cos(angle_rad),
                y=mag_demo*np.sin(angle_rad)+0.15,
                text="b", font=dict(color="#E24B4A",size=14), showarrow=False)
            # projection of b onto a
            proj = mag_demo*np.cos(angle_rad)
            fig.add_shape(type="line", x0=proj,y0=0,
                x1=mag_demo*np.cos(angle_rad), y1=mag_demo*np.sin(angle_rad),
                line=dict(color="#EF9F27",width=1.5,dash="dot"))
            fig.add_annotation(x=proj/2, y=-0.2, text=f"projection = {proj:.2f}",
                showarrow=False, font=dict(color="#EF9F27",size=11))
            fig.update_layout(xaxis=dict(range=[-0.5,mag_demo+0.5],zeroline=True),
                yaxis=dict(range=[-0.5,mag_demo+0.5],scaleanchor='x',scaleratio=1,zeroline=True),
                height=320, plot_bgcolor='white', showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.metric("Dot product a·b", f"{dp:.3f}")
        if angle_deg < 90:   st.success(f"Angle < 90° → positive dot product → vectors point in similar direction")
        elif angle_deg == 90: st.info("Angle = 90° → dot product = 0 → vectors are orthogonal (perpendicular)")
        else:                 st.warning(f"Angle > 90° → negative dot product → vectors point in opposite directions")
        st.markdown("> **In ML:** cosine similarity between two embedding vectors uses exactly this — dot product divided by magnitudes.")

    with tab3:
        st.markdown("### Common vector operations")
        col1, col2 = st.columns(2)
        v = np.array([ax, ay])
        u = np.array([bx, by])
        with col1:
            st.markdown("**Scalar multiplication**")
            scalar = st.slider("scalar s", -3.0, 3.0, 2.0, step=0.1)
            sv = scalar * v
            st.markdown(f'<div class="formula-box">s·a = {scalar}·[{ax},{ay}] = [{sv[0]:.2f},{sv[1]:.2f}]</div>', unsafe_allow_html=True)
            st.markdown("**Unit vector (normalise)**")
            unit = v / (np.linalg.norm(v) + 1e-9)
            st.markdown(f'<div class="formula-box">â = a/|a| = [{unit[0]:.3f},{unit[1]:.3f}]  |â|={np.linalg.norm(unit):.3f}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**L1 and L2 norms**")
            l1 = np.sum(np.abs(v)); l2 = np.linalg.norm(v)
            st.markdown(f'<div class="formula-box">‖a‖₁ = |{ax}|+|{ay}| = {l1:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="formula-box">‖a‖₂ = √({ax}²+{ay}²) = {l2:.3f}</div>', unsafe_allow_html=True)
            st.markdown("**Cross product magnitude (2D)**")
            cross = abs(ax*by - ay*bx)
            st.markdown(f'<div class="formula-box">|a×b| = |{ax}·{by} − {ay}·{bx}| = {cross:.3f}</div>', unsafe_allow_html=True)
            st.caption("= area of parallelogram formed by a and b")


# ═══════════════════════════════════════════════════════════════════════════
# MATRIX OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "matrix_ops":
    st.title("🔲 Matrix Operations")
    st.markdown("""
    <div class="concept-card">
    A <b>matrix</b> is a 2D array of numbers. Every layer in a neural network applies a
    matrix multiplication. Understanding what matrices <em>do geometrically</em> builds
    strong intuition for weights, transformations and embeddings.
    </div>
    """, unsafe_allow_html=True)

    tab0, tab1, tab2, tab3 = st.tabs(["Addition & subtraction", "Multiplication", "Geometric transformation", "Transpose, determinant & norms"])

    with tab0:
        st.markdown("### Matrix addition and subtraction")
        st.markdown("""
        Matrices add and subtract **element-by-element** — both matrices must have the
        same shape. This is the simplest matrix operation, but it's everywhere: combining
        gradients, accumulating updates, residual connections.
        """)
        st.markdown('<div class="formula-box">(A ± B)ᵢⱼ = Aᵢⱼ ± Bᵢⱼ</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Matrix A (2×2)**")
            Aas = np.array([[st.number_input(f"A[{i+1},{j+1}]", value=float([[2,1],[0,3]][i][j]),
                step=0.5, key=f"as_A{i}{j}") for j in range(2)] for i in range(2)])
            st.markdown("**Matrix B (2×2)**")
            Bas = np.array([[st.number_input(f"B[{i+1},{j+1}]", value=float([[1,4],[2,1]][i][j]),
                step=0.5, key=f"as_B{i}{j}") for j in range(2)] for i in range(2)])
            op = st.radio("Operation", ["A + B", "A − B"], horizontal=True)

        with col2:
            result = Aas + Bas if op == "A + B" else Aas - Bas
            symbol = "+" if op == "A + B" else "−"

            st.markdown(f"**Result  A {symbol} B**")
            st.markdown(f'<div class="formula-box">'
                f'[ {result[0,0]:.1f} , {result[0,1]:.1f} ]<br>'
                f'[ {result[1,0]:.1f} , {result[1,1]:.1f} ]'
                f'</div>', unsafe_allow_html=True)

            st.markdown("**Element-by-element breakdown:**")
            for i in range(2):
                for j in range(2):
                    st.markdown(f"({op[0]}{symbol}{op[0]})[{i+1},{j+1}] = "
                                f"{Aas[i,j]:.1f} {symbol} {Bas[i,j]:.1f} = **{result[i,j]:.2f}**")

            st.markdown("---")
            st.markdown("**Note:** addition/subtraction require identical shapes — "
                        "unlike multiplication, there's no inner-dimension rule to satisfy.")
            st.caption("In ML: gradient accumulation (summing gradients across a batch), "
                      "residual/skip connections (x + F(x)), and weight updates "
                      "(W ← W − η·∇W) are all matrix addition/subtraction in disguise.")


    with tab1:
        st.markdown("### Matrix × Matrix and Matrix × Vector")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Matrix A (2×3)**")
            A = np.array([[st.number_input(f"A[{i+1},{j+1}]", value=float([[1,2,0],[0,1,3]][i][j]),
                step=0.5, key=f"A{i}{j}") for j in range(3)] for i in range(2)])
            st.markdown("**Matrix B (3×2)**")
            B = np.array([[st.number_input(f"B[{i+1},{j+1}]", value=float([[1,0],[2,1],[0,3]][i][j]),
                step=0.5, key=f"B{i}{j}") for j in range(2)] for i in range(3)])

        with col2:
            C = A @ B
            st.markdown("**Result C = A × B  (2×2)**")
            st.markdown(f'<div class="formula-box">'
                f'[ {C[0,0]:.1f} , {C[0,1]:.1f} ]<br>'
                f'[ {C[1,0]:.1f} , {C[1,1]:.1f} ]'
                f'</div>', unsafe_allow_html=True)
            st.markdown("**How C[i,j] is computed — dot product of row i of A with column j of B:**")
            for i in range(2):
                for j in range(2):
                    terms = " + ".join([f"{A[i,k]:.1f}·{B[k,j]:.1f}" for k in range(3)])
                    st.markdown(f"C[{i+1},{j+1}] = {terms} = **{C[i,j]:.2f}**")

            st.markdown("**Matrix × vector** (A applied to a 3D input):")
            vx = st.number_input("v₁", value=1.0, step=0.5, key="mv1")
            vy = st.number_input("v₂", value=0.0, step=0.5, key="mv2")
            vz = st.number_input("v₃", value=2.0, step=0.5, key="mv3")
            v_in = np.array([vx, vy, vz])
            v_out = A @ v_in
            st.markdown(f'<div class="formula-box">A · v = [{v_out[0]:.2f}, {v_out[1]:.2f}]</div>',
                unsafe_allow_html=True)
            st.caption(f"Input: 3D → Output: 2D  (A projects the vector into a lower-dimensional space)")

    with tab2:
        st.markdown("### What a 2×2 matrix does to space")
        st.markdown("Every 2×2 matrix is a geometric transformation — rotation, scaling, shearing or reflection.")
        col1, col2 = st.columns([1, 2])
        with col1:
            preset = st.selectbox("Preset transformation", [
                "Custom", "Rotate 45°", "Scale x2", "Shear", "Reflection (y-axis)", "Stretch x, squash y"])
            presets = {
                "Rotate 45°":            [[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]],
                "Scale x2":              [[2,0],[0,2]],
                "Shear":                 [[1,1],[0,1]],
                "Reflection (y-axis)":   [[-1,0],[0,1]],
                "Stretch x, squash y":   [[2,0],[0,0.5]],
            }
            if preset != "Custom":
                default = presets[preset]
            else:
                default = [[1,0],[0,1]]
            m00 = st.number_input("M[1,1]", value=float(default[0][0]), step=0.1, key="m00")
            m01 = st.number_input("M[1,2]", value=float(default[0][1]), step=0.1, key="m01")
            m10 = st.number_input("M[2,1]", value=float(default[1][0]), step=0.1, key="m10")
            m11 = st.number_input("M[2,2]", value=float(default[1][1]), step=0.1, key="m11")
            M = np.array([[m00,m01],[m10,m11]])
            det = np.linalg.det(M)
            st.metric("Determinant", f"{det:.3f}")
            st.caption("det=0 → matrix collapses space (not invertible). |det|>1 → expands. |det|<1 → shrinks.")

        with col2:
            # unit square and transformed square
            square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]).T
            t_square = M @ square
            # basis vectors
            e1 = M @ np.array([1,0])
            e2 = M @ np.array([0,1])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=square[0], y=square[1], fill='toself',
                fillcolor='rgba(83,74,183,0.15)', line=dict(color='#534AB7',width=2),
                name='Original unit square'))
            fig.add_trace(go.Scatter(x=t_square[0], y=t_square[1], fill='toself',
                fillcolor='rgba(226,75,74,0.15)', line=dict(color='#E24B4A',width=2),
                name='Transformed square'))
            # original basis
            for v, col, lbl in [([1,0],'#534AB7','e₁'),([0,1],'#1D9E75','e₂')]:
                fig.add_shape(type='line',x0=0,y0=0,x1=v[0],y1=v[1],
                    line=dict(color=col,width=2,dash='dash'))
            # transformed basis
            for v, col, lbl in [(e1,'#E24B4A','Me₁'),(e2,'#EF9F27','Me₂')]:
                fig.add_shape(type='line',x0=0,y0=0,x1=v[0],y1=v[1],
                    line=dict(color=col,width=2.5))
                fig.add_annotation(x=v[0],y=v[1],text=lbl,showarrow=True,
                    arrowhead=2,arrowcolor=col,font=dict(color=col,size=11))
            lim = max(3, float(np.abs(t_square).max())+0.5)
            fig.update_layout(xaxis=dict(range=[-lim,lim],zeroline=True),
                yaxis=dict(range=[-lim,lim],zeroline=True,scaleanchor='x',scaleratio=1),
                height=420, plot_bgcolor='white',
                legend=dict(orientation='h',y=1.1),
                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Transpose and key properties")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transpose:** flip rows and columns")
            Tm = np.array([[1,2,3],[4,5,6]])
            st.markdown(f'<div class="formula-box">'
                f'A = [[1,2,3],[4,5,6]]<br>Aᵀ = [[1,4],[2,5],[3,6]]</div>',
                unsafe_allow_html=True)
            st.markdown("**Shape rule for multiplication:**")
            st.markdown(f'<div class="formula-box">(m×n) · (n×p) → (m×p)</div>', unsafe_allow_html=True)
            st.caption("The inner dimensions must match. This is why layer shapes in neural networks must align.")
        with col2:
            st.markdown("**Identity matrix:** A·I = A")
            st.markdown(f'<div class="formula-box">I = [[1,0],[0,1]]</div>', unsafe_allow_html=True)
            st.markdown("**Inverse:** A·A⁻¹ = I  (only if det ≠ 0)")
            M2 = np.array([[m00,m01],[m10,m11]])
            if abs(np.linalg.det(M2)) > 1e-6:
                inv = np.linalg.inv(M2)
                st.markdown(f'<div class="formula-box">M⁻¹ = [[{inv[0,0]:.2f},{inv[0,1]:.2f}],<br>[{inv[1,0]:.2f},{inv[1,1]:.2f}]]</div>',
                    unsafe_allow_html=True)
            else:
                st.error("This matrix is singular (det=0) — no inverse exists.")

        st.markdown("---")
        st.markdown("### Determinant — how much a matrix scales area/volume")
        st.markdown("""
        The **determinant** measures how a matrix scales area (2D) or volume (3D), with
        sign indicating whether orientation flips. It's the single number that tells you
        whether a matrix is invertible at all.
        """)
        st.markdown('<div class="formula-box">det([[a,b],[c,d]]) = ad − bc</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            d00 = st.number_input("D[1,1]", value=2.0, step=0.5, key="det00")
            d01 = st.number_input("D[1,2]", value=1.0, step=0.5, key="det01")
            d10 = st.number_input("D[2,1]", value=0.0, step=0.5, key="det10")
            d11 = st.number_input("D[2,2]", value=1.5, step=0.5, key="det11")
            Dm = np.array([[d00,d01],[d10,d11]])
            det_val = np.linalg.det(Dm)

            st.markdown(f'<div class="formula-box">det(D) = {d00}·{d11} − {d01}·{d10} = {det_val:.3f}</div>',
                unsafe_allow_html=True)

            if abs(det_val) < 1e-6:
                st.error("det = 0 → matrix collapses space onto a line or point. Not invertible (singular).")
            elif det_val < 0:
                st.warning(f"det < 0 ({det_val:.2f}) → orientation flips (reflection) and area scales by {abs(det_val):.2f}×")
            elif det_val < 1:
                st.info(f"0 < det < 1 ({det_val:.2f}) → area shrinks to {det_val:.2f}× original")
            else:
                st.success(f"det > 1 ({det_val:.2f}) → area expands to {det_val:.2f}× original")

        with col2:
            unit_sq = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]).T
            t_sq = Dm @ unit_sq
            area = abs(det_val)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=unit_sq[0], y=unit_sq[1], fill='toself',
                fillcolor='rgba(83,74,183,0.15)', line=dict(color='#534AB7', width=2),
                name='Unit square (area=1)'))
            fig.add_trace(go.Scatter(x=t_sq[0], y=t_sq[1], fill='toself',
                fillcolor='rgba(226,75,74,0.15)', line=dict(color='#E24B4A', width=2),
                name=f'D × square (area={area:.2f})'))
            lim = max(3, float(np.abs(t_sq).max()) + 0.5)
            fig.update_layout(
                xaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc"),
                yaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc", scaleanchor='x'),
                height=380, plot_bgcolor='white',
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The determinant is exactly the ratio of transformed area to original area "
                      "(unsigned). Sign tells you if the square also got flipped.")

        st.markdown("""
        **In ML:** determinant = 0 means a weight matrix has collapsed information into a
        lower-dimensional subspace (rank deficiency) — a sign of redundant features or a
        degenerate transformation. Normalizing flows use the determinant of the Jacobian
        to track how probability density changes under a transformation.
        """)

        st.markdown("---")
        st.markdown("### Frobenius norm — the magnitude of a whole matrix")
        st.markdown("""
        The **Frobenius norm** generalises the Euclidean (L2) vector norm to matrices —
        treat every entry as if it were one long vector and take its L2 norm.
        """)
        st.markdown('<div class="formula-box">‖A‖_F = √(Σᵢ Σⱼ Aᵢⱼ²)  =  √(trace(AᵀA))</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            f00 = st.number_input("F[1,1]", value=3.0, step=0.5, key="frob00")
            f01 = st.number_input("F[1,2]", value=1.0, step=0.5, key="frob01")
            f10 = st.number_input("F[2,1]", value=0.0, step=0.5, key="frob10")
            f11 = st.number_input("F[2,2]", value=4.0, step=0.5, key="frob11")
            Fm = np.array([[f00,f01],[f10,f11]])
            frob = np.linalg.norm(Fm, 'fro')

            terms = " + ".join([f"{Fm[i,j]:.1f}²" for i in range(2) for j in range(2)])
            st.markdown(f'<div class="formula-box">‖F‖_F = √({terms})<br>= √({np.sum(Fm**2):.2f}) = {frob:.3f}</div>',
                unsafe_allow_html=True)

        with col2:
            st.markdown("**Used for:**")
            st.markdown("""
            - **Weight decay / L2 regularization** on whole weight matrices: λ‖W‖_F²
            - **Matrix approximation error** — how close is à to A? Use ‖A − Ã‖_F
            - **Low-rank approximation (SVD)** — the Eckart-Young theorem says the best
              rank-k approximation under Frobenius norm comes from truncating the SVD
            - **Gradient clipping** — clip the Frobenius norm of a weight gradient matrix
              to prevent exploding gradients
            """)
            st.markdown('<div class="formula-box">Loss + λ‖W‖_F²<br><small>matrix analogue of L2 weight regularization</small></div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DERIVATIVE
# ═══════════════════════════════════════════════════════════════════════════
elif section == "derivative":
    st.title("📐 Derivative")
    st.markdown("""
    <div class="concept-card">
    The <b>derivative</b> f'(x) is the instantaneous rate of change of f at x —
    the slope of the tangent line at that point. It answers: <em>"if x increases by a tiny
    amount, how much does f change?"</em>. This is the engine behind every gradient update in ML.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">f\'(x) = lim<sub>h→0</sub> [f(x+h) − f(x)] / h</div>',
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Tangent line explorer", "Numerical vs analytical", "Common rules"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            fn_choice = st.selectbox("Function", ["x²", "x³", "sin(x)", "cos(x)", "eˣ", "ln(x)", "x² + 2x − 3"])
            x0 = st.slider("Point x₀", -3.0, 3.0, 1.0, step=0.05)
            show_limit = st.checkbox("Show secant → tangent (h shrinks)", value=False)
            h_val = st.select_slider("h (secant step)", [2.0,1.0,0.5,0.2,0.1,0.01], value=0.5) if show_limit else 0.01

        fn_map = {
            "x²":        (lambda x: x**2,          lambda x: 2*x,           "2x"),
            "x³":        (lambda x: x**3,          lambda x: 3*x**2,        "3x²"),
            "sin(x)":    (lambda x: np.sin(x),     lambda x: np.cos(x),     "cos(x)"),
            "cos(x)":    (lambda x: np.cos(x),     lambda x: -np.sin(x),    "−sin(x)"),
            "eˣ":        (lambda x: np.exp(np.clip(x,-10,5)), lambda x: np.exp(np.clip(x,-10,5)), "eˣ"),
            "ln(x)":     (lambda x: np.log(np.abs(x)+1e-9), lambda x: 1/(x+1e-9), "1/x"),
            "x² + 2x − 3": (lambda x: x**2+2*x-3, lambda x: 2*x+2,         "2x + 2"),
        }
        f, df, df_str = fn_map[fn_choice]

        x_range = np.linspace(-3.5, 3.5, 400)
        y_range = np.array([f(xi) for xi in x_range])
        slope = df(x0)
        y0_val = f(x0)
        tangent_y = slope*(x_range - x0) + y0_val

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=np.clip(y_range,-8,8), name=fn_choice,
                line=dict(color='#534AB7', width=2.5)))
            fig.add_trace(go.Scatter(x=x_range, y=np.clip(tangent_y,-8,8),
                name=f"Tangent at x={x0:.2f}", line=dict(color='#E24B4A', width=2, dash='dash')))
            fig.add_scatter(x=[x0], y=[y0_val], mode='markers',
                marker=dict(color='#E24B4A', size=12), name='Point', showlegend=False)
            if show_limit:
                x1_s = x0 + h_val
                y1_s = f(x1_s)
                sec_slope = (y1_s - y0_val) / (h_val + 1e-12)
                sec_y = sec_slope*(x_range - x0) + y0_val
                fig.add_trace(go.Scatter(x=x_range, y=np.clip(sec_y,-8,8),
                    name=f"Secant (h={h_val})", line=dict(color='#EF9F27', width=2)))
                fig.add_scatter(x=[x1_s], y=[y1_s], mode='markers',
                    marker=dict(color='#EF9F27', size=10), showlegend=False)
            fig.update_layout(xaxis_title="x", yaxis_title="y",
                yaxis_range=[-8, 8], height=400, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="formula-box">f(x) = {fn_choice} &nbsp;→&nbsp; f\'(x) = {df_str} &nbsp;→&nbsp; f\'({x0:.2f}) = {slope:.4f}</div>',
            unsafe_allow_html=True)
        st.caption("The tangent slope equals the derivative. As h→0, the orange secant converges to the red tangent.")

    with tab2:
        st.markdown("### Numerical approximation vs analytical formula")
        col1, col2 = st.columns([1, 2])
        with col1:
            fn2 = st.selectbox("Function", ["x²", "sin(x)", "eˣ"], key="d_fn2")
            x_num = st.slider("x", -3.0, 3.0, 1.5, step=0.1, key="d_xnum")
            h_num = st.select_slider("h (step size)", [1.0,0.5,0.1,0.01,0.001,0.0001], value=0.1)
        f2, df2, df2_str = fn_map[fn2]
        analytical = df2(x_num)
        forward  = (f2(x_num+h_num) - f2(x_num)) / h_num
        central  = (f2(x_num+h_num) - f2(x_num-h_num)) / (2*h_num)
        with col2:
            st.markdown(f'<div class="formula-box">Analytical f\'(x) = {df2_str} = {analytical:.6f}</div>',
                unsafe_allow_html=True)
            import pandas as pd
            df_num = pd.DataFrame({
                "Method": ["Forward difference", "Central difference", "Analytical"],
                "Formula": [f"[f(x+h)−f(x)]/h", f"[f(x+h)−f(x−h)]/2h", f"{df2_str}"],
                "Value": [f"{forward:.6f}", f"{central:.6f}", f"{analytical:.6f}"],
                "Error": [f"{abs(forward-analytical):.2e}", f"{abs(central-analytical):.2e}", "0"],
            })
            st.dataframe(df_num, use_container_width=True, hide_index=True)
        st.caption("Central difference is more accurate for the same h — error is O(h²) vs O(h) for forward.")

    with tab3:
        st.markdown("### Key differentiation rules")
        rules = [
            ("Power rule",     "f(x) = xⁿ",       "f'(x) = n·xⁿ⁻¹",              "x³ → 3x²"),
            ("Sum rule",       "f+g",               "(f+g)' = f'+g'",               "x²+sin(x) → 2x+cos(x)"),
            ("Product rule",   "f·g",               "(fg)' = f'g + fg'",            "x·sin(x) → sin(x)+x·cos(x)"),
            ("Chain rule",     "f(g(x))",           "f'(g(x))·g'(x)",               "sin(x²) → cos(x²)·2x"),
            ("Exponential",    "eˣ",                "eˣ",                           "e³ˣ → 3e³ˣ"),
            ("Logarithm",      "ln(x)",             "1/x",                          "ln(2x) → 1/x"),
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rules, columns=["Rule","f(x)","f'(x)","Example"]),
            use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PARTIAL DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════
elif section == "partial_deriv":
    st.title("∂ Partial Derivatives")
    st.markdown("""
    <div class="concept-card">
    A <b>partial derivative</b> ∂f/∂x measures how f changes when only one variable moves,
    holding all others fixed. The vector of all partial derivatives is the <b>gradient</b> ∇f —
    the direction of steepest ascent used in every ML optimiser.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">∇f(x,y) = [∂f/∂x , ∂f/∂y]</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        fn_p = st.selectbox("Function f(x,y)", [
            "x² + y²", "sin(x)·cos(y)", "x²·y + y³", "eˣ⁺ʸ", "x² − y²"])
        x_p = st.slider("x₀", -2.0, 2.0, 1.0, step=0.1, key="pd_x")
        y_p = st.slider("y₀", -2.0, 2.0, 0.5, step=0.1, key="pd_y")
        show_grad_arrows = st.checkbox("Show gradient arrows on surface", value=True)

    fn_p_map = {
        "x² + y²":      (lambda x,y: x**2+y**2,     lambda x,y: 2*x,             lambda x,y: 2*y,           "2x","2y"),
        "sin(x)·cos(y)":(lambda x,y: np.sin(x)*np.cos(y), lambda x,y: np.cos(x)*np.cos(y), lambda x,y: -np.sin(x)*np.sin(y), "cos(x)cos(y)","-sin(x)sin(y)"),
        "x²·y + y³":    (lambda x,y: x**2*y+y**3,   lambda x,y: 2*x*y,          lambda x,y: x**2+3*y**2,   "2xy","x²+3y²"),
        "eˣ⁺ʸ":         (lambda x,y: np.exp(np.clip(x+y,-10,5)), lambda x,y: np.exp(np.clip(x+y,-10,5)), lambda x,y: np.exp(np.clip(x+y,-10,5)), "eˣ⁺ʸ","eˣ⁺ʸ"),
        "x² − y²":      (lambda x,y: x**2-y**2,     lambda x,y: 2*x,             lambda x,y: -2*y,          "2x","-2y"),
    }
    f_p, dfdx_fn, dfdy_fn, dfdx_str, dfdy_str = fn_p_map[fn_p]
    dfdx_val = dfdx_fn(x_p, y_p)
    dfdy_val = dfdy_fn(x_p, y_p)
    f_val = f_p(x_p, y_p)

    with col2:
        xs = np.linspace(-2.5, 2.5, 60)
        ys = np.linspace(-2.5, 2.5, 60)
        X, Y = np.meshgrid(xs, ys)
        Z = np.clip(f_p(X, Y), -10, 10)
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='RdPu',
            opacity=0.85, showscale=False)])
        # mark the point
        fig.add_trace(go.Scatter3d(x=[x_p], y=[y_p], z=[f_val], mode='markers',
            marker=dict(color='#E24B4A', size=6), name='(x₀,y₀)'))
        # gradient arrow in xy-plane (projected)
        grad_len = 0.5
        fig.add_trace(go.Scatter3d(
            x=[x_p, x_p + grad_len*dfdx_val/(np.sqrt(dfdx_val**2+dfdy_val**2)+1e-9)],
            y=[y_p, y_p + grad_len*dfdy_val/(np.sqrt(dfdx_val**2+dfdy_val**2)+1e-9)],
            z=[f_val, f_val], mode='lines',
            line=dict(color='#EF9F27', width=6), name='Gradient direction'))
        fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
            height=430, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'<div class="formula-box">'
        f'∂f/∂x = {dfdx_str} = {dfdx_val:.4f} &nbsp;&nbsp; ∂f/∂y = {dfdy_str} = {dfdy_val:.4f}'
        f'</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("f(x₀,y₀)", f"{f_val:.4f}")
    c2.metric("∂f/∂x at point", f"{dfdx_val:.4f}")
    c3.metric("∂f/∂y at point", f"{dfdy_val:.4f}")
    st.markdown(f"**Gradient vector:** ∇f = [{dfdx_val:.4f}, {dfdy_val:.4f}] — points in direction of steepest ascent. "
        f"Gradient descent moves in the **opposite** direction: −∇f.")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRAL
# ═══════════════════════════════════════════════════════════════════════════
elif section == "integral":
    st.title("∫ Integral")
    st.markdown("""
    <div class="concept-card">
    The <b>integral</b> accumulates the total area between a function and the x-axis.
    In ML it appears in probability (area under a PDF = 1), expected values,
    and understanding continuous loss landscapes.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">∫ₐᵇ f(x) dx = lim<sub>n→∞</sub> Σ f(xᵢ)·Δx</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Riemann sum visualisation", "Common integrals"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            fn_i = st.selectbox("Function", ["x²", "sin(x)", "eˣ", "x³−2x", "cos(x)"])
            a_i = st.slider("Lower limit a", -3.0, 2.9, 0.0, step=0.1)
            b_i = st.slider("Upper limit b", a_i+0.1, 3.0, 2.0, step=0.1)
            n_rect = st.slider("Number of rectangles n", 2, 100, 10)
            method = st.radio("Method", ["Left", "Right", "Midpoint"], horizontal=True)

        fn_i_map = {
            "x²":      (lambda x: x**2,            "x³/3"),
            "sin(x)":  (lambda x: np.sin(x),       "−cos(x)"),
            "eˣ":      (lambda x: np.exp(np.clip(x,-10,4)), "eˣ"),
            "x³−2x":   (lambda x: x**3 - 2*x,      "x⁴/4 − x²"),
            "cos(x)":  (lambda x: np.cos(x),       "sin(x)"),
        }
        f_i, antideriv_str = fn_i_map[fn_i]

        x_plot = np.linspace(a_i-0.3, b_i+0.3, 400)
        y_plot = np.array([f_i(xi) for xi in x_plot])

        dx = (b_i - a_i) / n_rect
        if method == "Left":     xs_rect = np.linspace(a_i, b_i-dx, n_rect)
        elif method == "Right":  xs_rect = np.linspace(a_i+dx, b_i, n_rect)
        else:                    xs_rect = np.linspace(a_i+dx/2, b_i-dx/2, n_rect)
        heights = np.array([f_i(xi) for xi in xs_rect])
        riemann_sum = float(np.sum(heights) * dx)

        # numerical true integral
        x_fine = np.linspace(a_i, b_i, 2000)
        true_integral = float(np.trapz([f_i(xi) for xi in x_fine], x_fine))

        with col2:
            fig = go.Figure()
            for xi, hi in zip(xs_rect, heights):
                fig.add_shape(type='rect',
                    x0=xi-dx/2 if method=="Midpoint" else (xi if method=="Left" else xi-dx),
                    y0=0, x1=xi+dx/2 if method=="Midpoint" else (xi+dx if method=="Left" else xi),
                    y1=hi,
                    fillcolor='rgba(83,74,183,0.35)', line=dict(color='#534AB7', width=0.5))
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name=fn_i,
                line=dict(color='#E24B4A', width=2.5)))
            fig.add_vline(x=a_i, line_dash='dot', line_color='gray')
            fig.add_vline(x=b_i, line_dash='dot', line_color='gray')
            fig.update_layout(xaxis_title="x", yaxis_title="f(x)",
                height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Riemann sum", f"{riemann_sum:.5f}")
        c2.metric("True integral", f"{true_integral:.5f}")
        c3.metric("Error", f"{abs(riemann_sum-true_integral):.2e}")
        st.caption(f"Antiderivative: F(x) = {antideriv_str}  →  F({b_i:.1f}) − F({a_i:.1f}) = {true_integral:.5f}")

    with tab2:
        import pandas as pd
        integrals = [
            ("xⁿ  (n≠−1)",   "xⁿ⁺¹/(n+1) + C",  "Power rule — every polynomial term"),
            ("1/x",           "ln|x| + C",         "Appears in log-loss, entropy"),
            ("eˣ",            "eˣ + C",            "Its own antiderivative — unique property"),
            ("sin(x)",        "−cos(x) + C",       "Oscillating functions, Fourier"),
            ("cos(x)",        "sin(x) + C",        "Oscillating functions"),
            ("1/(1+x²)",      "arctan(x) + C",     "Probability distributions"),
        ]
        st.dataframe(pd.DataFrame(integrals, columns=["f(x)","∫f(x)dx","Where it appears"]),
            use_container_width=True, hide_index=True)
        st.markdown("**Fundamental Theorem of Calculus** connects derivatives and integrals:")
        st.markdown('<div class="formula-box">d/dx ∫ₐˣ f(t)dt = f(x)</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHAIN RULE
# ═══════════════════════════════════════════════════════════════════════════
elif section == "chain_rule":
    st.title("🔗 Chain Rule")
    st.markdown("""
    <div class="concept-card">
    The <b>chain rule</b> tells us how to differentiate a <em>composition</em> of functions:
    if y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). This is the mathematical heart of
    backpropagation — every gradient in a neural network is computed via repeated chain rule.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">dy/dx = (dy/du) · (du/dx)&nbsp;&nbsp; where u = g(x)</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Step-by-step decomposition", "Chain rule in a neural network"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            outer = st.selectbox("Outer function f(u)", ["u²", "sin(u)", "eᵘ", "ln(u)", "√u"])
            inner = st.selectbox("Inner function g(x)", ["2x+1", "x²", "cos(x)", "x³−x"])
            x_cr  = st.slider("x₀", -2.0, 2.0, 1.0, step=0.1, key="cr_x")

        outer_map = {
            "u²":   (lambda u: u**2,                          lambda u: 2*u,                    "2u"),
            "sin(u)":(lambda u: np.sin(u),                    lambda u: np.cos(u),              "cos(u)"),
            "eᵘ":   (lambda u: np.exp(np.clip(u,-10,5)),     lambda u: np.exp(np.clip(u,-10,5)),"eᵘ"),
            "ln(u)":(lambda u: np.log(np.abs(u)+1e-9),       lambda u: 1/(u+1e-9),             "1/u"),
            "√u":   (lambda u: np.sqrt(np.abs(u)),           lambda u: 0.5/np.sqrt(np.abs(u)+1e-9),"1/(2√u)"),
        }
        inner_map = {
            "2x+1":   (lambda x: 2*x+1,   lambda x: np.full_like(np.array([x],dtype=float),2.0)[0],  "2"),
            "x²":     (lambda x: x**2,    lambda x: 2*x,           "2x"),
            "cos(x)": (lambda x: np.cos(x), lambda x: -np.sin(x), "−sin(x)"),
            "x³−x":   (lambda x: x**3-x,  lambda x: 3*x**2-1,     "3x²−1"),
        }
        f_out, df_out, df_out_str = outer_map[outer]
        g_in,  dg_in,  dg_in_str  = inner_map[inner]

        u_val   = g_in(x_cr)
        y_val   = f_out(u_val)
        du_dx   = dg_in(x_cr)
        dy_du   = df_out(u_val)
        dy_dx   = dy_du * du_dx

        with col2:
            st.markdown("**Decomposition at x₀ = {:.2f}**".format(x_cr))
            st.code(f"""
Step 1 — Evaluate inner function g(x):
   u = g({x_cr:.2f}) = {inner} = {u_val:.4f}

Step 2 — Evaluate outer function f(u):
   y = f(u) = f({u_val:.4f}) = {outer} = {y_val:.4f}

Step 3 — Derivative of inner:
   du/dx = g'(x) = {dg_in_str} = {du_dx:.4f}

Step 4 — Derivative of outer (at u):
   dy/du = f'(u) = {df_out_str} = {dy_du:.4f}

Step 5 — Chain rule:
   dy/dx = (dy/du)·(du/dx)
         = {dy_du:.4f} × {du_dx:.4f}
         = {dy_dx:.4f}
""", language="text")

        # plot f(g(x)) and its derivative
        x_range_cr = np.linspace(-2.5, 2.5, 300)
        y_comp = np.array([f_out(g_in(xi)) for xi in x_range_cr])
        dy_comp = np.array([df_out(g_in(xi))*dg_in(xi) for xi in x_range_cr])
        tangent_cr = dy_dx*(x_range_cr - x_cr) + y_val

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[f"f(g(x)) = {outer}({inner})", "Derivative via chain rule"])
        fig.add_trace(go.Scatter(x=x_range_cr, y=np.clip(y_comp,-8,8), name='f(g(x))',
            line=dict(color='#534AB7',width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_range_cr, y=np.clip(tangent_cr,-8,8),
            name='Tangent', line=dict(color='#E24B4A',width=1.5,dash='dash')), row=1, col=1)
        fig.add_scatter(x=[x_cr], y=[y_val], mode='markers',
            marker=dict(color='#E24B4A',size=10), showlegend=False, row=1, col=1)
        fig.add_trace(go.Scatter(x=x_range_cr, y=np.clip(dy_comp,-8,8), name="d/dx f(g(x))",
            line=dict(color='#1D9E75',width=2.5)), row=1, col=2)
        fig.add_scatter(x=[x_cr], y=[dy_dx], mode='markers',
            marker=dict(color='#1D9E75',size=10), showlegend=False, row=1, col=2)
        fig.update_xaxes(title_text="x"); fig.update_yaxes(title_text="y")
        fig.update_layout(height=360, legend=dict(orientation='h',y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Why the chain rule *is* backpropagation")
        st.markdown("""
        Consider a 3-layer network: input x → layer 1 → layer 2 → loss L.

        To update w₁ we need dL/dw₁. We never have a direct formula, but the chain rule gives:
        """)
        st.markdown('<div class="formula-box">dL/dw₁ = (dL/da₂)·(da₂/da₁)·(da₁/dw₁)</div>',
            unsafe_allow_html=True)
        st.markdown("""
        Each term is easy to compute locally at each layer.
        Backpropagation is just the chain rule applied efficiently from output → input,
        reusing intermediate values so nothing is computed twice.
        """)

        # simulate a tiny network chain rule numerically
        st.markdown("**Numerical demonstration — 3-layer chain:**")
        col1, col2 = st.columns([1, 2])
        with col1:
            x_nn = st.slider("Input x", -2.0, 2.0, 1.0, step=0.1, key="cr_nn_x")
            w1_c = st.slider("Weight w₁", -2.0, 2.0, 0.5, step=0.1, key="cr_w1")
            w2_c = st.slider("Weight w₂", -2.0, 2.0, 1.5, step=0.1, key="cr_w2")
            w3_c = st.slider("Weight w₃", -2.0, 2.0,-0.8, step=0.1, key="cr_w3")
            y_tgt= st.slider("Target y",  -2.0, 2.0, 1.0, step=0.1, key="cr_y")

        z1 = w1_c * x_nn;         a1 = np.tanh(z1)
        z2 = w2_c * a1;           a2 = np.tanh(z2)
        z3 = w3_c * a2;           y_hat_c = z3
        L  = 0.5*(y_hat_c - y_tgt)**2
        dL_dyhat = y_hat_c - y_tgt
        dyhat_dw3 = a2
        da2_dz2 = 1 - np.tanh(z2)**2
        dz2_dw2 = a1
        da1_dz1 = 1 - np.tanh(z1)**2
        dz1_dw1 = x_nn
        grad_w3 = dL_dyhat * dyhat_dw3
        grad_w2 = dL_dyhat * w3_c * da2_dz2 * dz2_dw2
        grad_w1 = dL_dyhat * w3_c * da2_dz2 * w2_c * da1_dz1 * dz1_dw1

        with col2:
            st.code(f"""
Forward:  x={x_nn} → z1={z1:.3f} → a1=tanh={a1:.3f}
          → z2={z2:.3f} → a2=tanh={a2:.3f}
          → z3=ŷ={y_hat_c:.3f}   Loss={L:.4f}

Backward (chain rule):
  dL/dw3 = {grad_w3:.4f}
  dL/dw2 = {grad_w2:.4f}
  dL/dw1 = {grad_w1:.4f}   ← gradient flows back through 3 layers
""", language="text")


# ═══════════════════════════════════════════════════════════════════════════
# DOT PRODUCT
# ═══════════════════════════════════════════════════════════════════════════
elif section == "dot_product":
    st.title("· Dot Product")
    st.markdown("""
    <div class="concept-card">
    The <b>dot product</b> multiplies corresponding elements of two vectors and sums them.
    It is the single most repeated operation in ML: every neuron computes a dot product,
    every attention score is a dot product, every cosine similarity is a normalised dot product.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">a · b = Σ aᵢbᵢ = |a||b|cos θ</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Interactive calculator", "Applications in ML"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            dim = st.slider("Vector dimension", 2, 6, 3)
            st.markdown("**Vector a**")
            a_dp = np.array([st.slider(f"a[{i+1}]", -3.0, 3.0,
                [1.0,2.0,-1.0,0.5,1.5,-0.5][i], step=0.1, key=f"dp_a{i}") for i in range(dim)])
            st.markdown("**Vector b**")
            b_dp = np.array([st.slider(f"b[{i+1}]", -3.0, 3.0,
                [2.0,-1.0,1.0,1.5,-0.5,0.5][i], step=0.1, key=f"dp_b{i}") for i in range(dim)])

        dp_val   = float(np.dot(a_dp, b_dp))
        mag_a_dp = float(np.linalg.norm(a_dp))
        mag_b_dp = float(np.linalg.norm(b_dp))
        cos_sim  = dp_val / (mag_a_dp * mag_b_dp + 1e-9)
        angle_dp = float(np.degrees(np.arccos(np.clip(cos_sim, -1, 1))))

        with col2:
            st.markdown("**Element-wise multiplication:**")
            terms = [f"({a_dp[i]:.1f})×({b_dp[i]:.1f}) = {a_dp[i]*b_dp[i]:.2f}" for i in range(dim)]
            for t in terms:
                st.markdown(f"&nbsp;&nbsp;&nbsp;{t}")
            st.markdown(f"**Sum = {dp_val:.4f}**")
            st.divider()
            c1,c2,c3,c4 = st.columns(2), st.columns(2)
            st.metric("|a|", f"{mag_a_dp:.3f}")
            st.metric("|b|", f"{mag_b_dp:.3f}")
            st.metric("a·b", f"{dp_val:.4f}")
            st.metric("cos θ", f"{cos_sim:.4f}")
            st.metric("Angle θ", f"{angle_dp:.1f}°")

            if cos_sim > 0.9:   st.success("Vectors are nearly identical in direction")
            elif cos_sim > 0.0: st.info("Vectors point in a similar direction")
            elif cos_sim == 0:  st.warning("Vectors are orthogonal (perpendicular)")
            else:               st.error("Vectors point in opposite directions")

        # bar chart of contributions
        fig = go.Figure()
        contribs = a_dp * b_dp
        colors_dp = ['#1D9E75' if c >= 0 else '#E24B4A' for c in contribs]
        fig.add_trace(go.Bar(x=[f"a[{i+1}]·b[{i+1}]" for i in range(dim)],
            y=contribs, marker_color=colors_dp, name='Element contribution'))
        fig.add_hline(y=0, line_color='gray', line_width=0.5)
        fig.update_layout(xaxis_title="Component", yaxis_title="aᵢ·bᵢ",
            title=f"Element contributions to dot product (sum = {dp_val:.3f})",
            height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("""
        ### Where the dot product appears in ML

        **1. Single neuron output:**
        """)
        st.markdown('<div class="formula-box">z = w · x + b = Σ wᵢxᵢ + b</div>', unsafe_allow_html=True)
        st.markdown("""
        **2. Cosine similarity** (used in embeddings, NLP, recommendation):
        """)
        st.markdown('<div class="formula-box">cos_sim(a,b) = a·b / (|a|·|b|)</div>', unsafe_allow_html=True)
        st.markdown("""
        **3. Attention score** (transformers):
        """)
        st.markdown('<div class="formula-box">score(Q,K) = Q·Kᵀ / √dₖ</div>', unsafe_allow_html=True)
        st.markdown("""
        **4. Projection** — how much of b lies along a:
        """)
        st.markdown('<div class="formula-box">proj_a(b) = (a·b / |a|²) · a</div>', unsafe_allow_html=True)

        st.markdown("### Interactive cosine similarity between two word embeddings (simulated)")
        col1, col2 = st.columns(2)
        with col1:
            np.random.seed(42)
            words = ["king","queen","man","woman","cat","dog","Paris","France"]
            w1_sel = st.selectbox("Word 1", words, index=0)
            w2_sel = st.selectbox("Word 2", words, index=1)
            # simple fake embeddings
            embeddings = {w: np.random.randn(8) for w in words}
            embeddings["queen"] = embeddings["king"] + np.array([0.1,-0.2,0.8,0.1,-0.1,0.2,-0.1,0.1])
            embeddings["woman"] = embeddings["man"]  + np.array([0.1,-0.2,0.8,0.1,-0.1,0.2,-0.1,0.1])
            embeddings["dog"]   = embeddings["cat"]  + np.random.randn(8)*0.3
            embeddings["France"]= embeddings["Paris"]+ np.random.randn(8)*0.3
            e1, e2 = embeddings[w1_sel], embeddings[w2_sel]
            cs = float(np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-9))
        with col2:
            st.metric(f"cos_sim({w1_sel}, {w2_sel})", f"{cs:.4f}")
            st.progress(float((cs+1)/2))
            st.caption("Simulated 8-dim embeddings. Related words cluster together.")


# ═══════════════════════════════════════════════════════════════════════════
# EIGENVALUES & EIGENVECTORS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "eigenvalues":
    st.title("λ Eigenvalues & Eigenvectors")
    st.markdown("""
    <div class="concept-card">
    An <b>eigenvector</b> of a matrix A is a special direction that doesn't rotate under A —
    it only stretches or shrinks. The <b>eigenvalue</b> λ tells you by how much.
    They appear in PCA (dimensionality reduction), understanding covariance, and stability analysis.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">A · v = λ · v</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Visual explorer", "PCA connection"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            preset_e = st.selectbox("Matrix preset", [
                "Custom", "Scaling", "Rotation 30°", "Shear", "Symmetric"])
            presets_e = {
                "Scaling":      [[3,0],[0,1.5]],
                "Rotation 30°": [[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]],
                "Shear":        [[1,2],[0,1]],
                "Symmetric":    [[3,1],[1,2]],
            }
            def_e = presets_e.get(preset_e, [[2,1],[1,2]])
            e00 = st.number_input("A[1,1]", value=float(def_e[0][0]), step=0.5, key="e00")
            e01 = st.number_input("A[1,2]", value=float(def_e[0][1]), step=0.5, key="e01")
            e10 = st.number_input("A[2,1]", value=float(def_e[1][0]), step=0.5, key="e10")
            e11 = st.number_input("A[2,2]", value=float(def_e[1][1]), step=0.5, key="e11")
            A_e = np.array([[e00,e01],[e10,e11]])

        try:
            eigenvalues, eigenvectors = np.linalg.eig(A_e)
            real_eigs = np.isreal(eigenvalues).all()
        except Exception:
            real_eigs = False

        with col2:
            fig = go.Figure()
            # show many vectors before/after transformation
            angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
            for ang in angles:
                v_orig = np.array([np.cos(ang), np.sin(ang)])
                v_trans = A_e @ v_orig
                fig.add_shape(type='line', x0=0,y0=0, x1=v_orig[0],y1=v_orig[1],
                    line=dict(color='rgba(83,74,183,0.3)',width=1))
                fig.add_shape(type='line', x0=0,y0=0, x1=v_trans[0],y1=v_trans[1],
                    line=dict(color='rgba(226,75,74,0.25)',width=1))

            # highlight eigenvectors
            if real_eigs:
                ev_colors = ['#EF9F27','#1D9E75']
                for i, (lam, ev, col_ev) in enumerate(zip(eigenvalues, eigenvectors.T, ev_colors)):
                    if np.isreal(lam):
                        ev_r = ev.real / (np.linalg.norm(ev.real)+1e-9)
                        fig.add_shape(type='line', x0=0,y0=0,
                            x1=ev_r[0], y1=ev_r[1],
                            line=dict(color=col_ev, width=3))
                        transformed = (A_e @ ev_r)
                        fig.add_shape(type='line', x0=0,y0=0,
                            x1=transformed[0], y1=transformed[1],
                            line=dict(color=col_ev, width=3, dash='dash'))
                        fig.add_annotation(x=ev_r[0]*1.2, y=ev_r[1]*1.2,
                            text=f"v{i+1} (λ={lam.real:.2f})",
                            font=dict(color=col_ev, size=12), showarrow=False)

            lim_e = max(3.0, float(np.abs(A_e).max())*1.5)
            fig.update_layout(
                xaxis=dict(range=[-lim_e,lim_e], zeroline=True, zerolinecolor='#ccc'),
                yaxis=dict(range=[-lim_e,lim_e], zeroline=True, zerolinecolor='#ccc',
                    scaleanchor='x', scaleratio=1),
                height=420, plot_bgcolor='white', showlegend=False,
                margin=dict(l=10,r=10,t=30,b=10),
                title="Blue=original vectors, Red=transformed. Gold/Green=eigenvectors (solid→dashed after A)")
            st.plotly_chart(fig, use_container_width=True)

        if real_eigs:
            c1,c2 = st.columns(2)
            for i, (lam, ev) in enumerate(zip(eigenvalues, eigenvectors.T)):
                col_c = [c1,c2][i]
                col_c.metric(f"λ{i+1}", f"{lam.real:.4f}")
                col_c.markdown(f"v{i+1} = [{ev[0].real:.3f}, {ev[1].real:.3f}]")
            st.markdown(f"**Trace** (sum of eigenvalues) = {np.trace(A_e):.3f} &nbsp;|&nbsp; "
                f"**Determinant** (product of eigenvalues) = {np.linalg.det(A_e):.3f}")
        else:
            st.warning("This matrix has complex eigenvalues (rotation without real eigenvectors).")

    with tab2:
        st.markdown("""
        ### PCA — Principal Component Analysis

        PCA finds the directions of maximum variance in data.
        These directions are exactly the **eigenvectors of the covariance matrix**,
        and the eigenvalues tell you how much variance each direction captures.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            n_pca = st.slider("Number of data points", 20, 200, 80)
            corr  = st.slider("Feature correlation", -0.95, 0.95, 0.8, step=0.05)
            seed_pca = st.slider("Seed", 0, 10, 1, key="pca_seed")

        np.random.seed(seed_pca)
        cov_mat = np.array([[1.0, corr],[corr, 1.0]])
        X_pca = np.random.multivariate_normal([0,0], cov_mat, n_pca)
        cov_emp = np.cov(X_pca.T)
        evals, evecs = np.linalg.eigh(cov_emp)
        # sort descending
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:,idx]
        var_explained = evals / evals.sum()

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_pca[:,0], y=X_pca[:,1], mode='markers',
                marker=dict(color='#534AB7', size=5, opacity=0.6), name='Data'))
            scale = 2.0
            for i, (val, vec, col_p) in enumerate(zip(evals, evecs.T, ['#E24B4A','#1D9E75'])):
                fig.add_shape(type='line',
                    x0=-vec[0]*np.sqrt(val)*scale, y0=-vec[1]*np.sqrt(val)*scale,
                    x1=vec[0]*np.sqrt(val)*scale,  y1=vec[1]*np.sqrt(val)*scale,
                    line=dict(color=col_p, width=3))
                fig.add_annotation(x=vec[0]*np.sqrt(val)*scale*1.1,
                    y=vec[1]*np.sqrt(val)*scale*1.1,
                    text=f"PC{i+1} ({var_explained[i]:.1%})",
                    font=dict(color=col_p, size=12), showarrow=False)
            fig.update_layout(xaxis_title="Feature 1", yaxis_title="Feature 2",
                xaxis=dict(range=[-4,4]), yaxis=dict(range=[-4,4],scaleanchor='x',scaleratio=1),
                height=400, plot_bgcolor='white',
                margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        c1.metric("PC1 explains", f"{var_explained[0]:.1%} of variance")
        c2.metric("PC2 explains", f"{var_explained[1]:.1%} of variance")
        st.caption("The principal components (eigenvectors) are the natural axes of the data cloud.")

# ═══════════════════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════════════════
elif section == "pca":
    st.title("🔍 PCA — Principal Component Analysis")
    st.markdown("""
    <div class="concept-card">
    PCA finds the directions of <b>maximum variance</b> in high-dimensional data and projects
    the data onto a lower-dimensional space, losing as little information as possible.
    It is one of the most widely used dimensionality reduction techniques in ML.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["2D → 1D intuition", "Multi-dimensional PCA", "Scree plot & variance"])

    with tab1:
        st.markdown("The first principal component is the direction that captures the most variance.")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_pca2 = st.slider("Points", 30, 200, 80, key="pca2_n")
            corr2  = st.slider("Feature correlation", -0.95, 0.95, 0.75, step=0.05, key="pca2_c")
            noise2 = st.slider("Noise", 0.05, 0.8, 0.2, step=0.05, key="pca2_ns")
            seed2  = st.slider("Seed", 0, 20, 3, key="pca2_s")
            show_proj = st.checkbox("Show projections onto PC1", value=True)

        np.random.seed(seed2)
        cov2 = np.array([[1.0, corr2],[corr2, 1.0]])
        X2 = np.random.multivariate_normal([0,0], cov2, n_pca2)
        X2 += np.random.randn(*X2.shape) * noise2

        # manual PCA
        X2c = X2 - X2.mean(axis=0)
        cov_e = np.cov(X2c.T)
        evals2, evecs2 = np.linalg.eigh(cov_e)
        idx2 = np.argsort(evals2)[::-1]
        evals2, evecs2 = evals2[idx2], evecs2[:, idx2]
        pc1 = evecs2[:, 0]
        var_exp = evals2 / evals2.sum()

        # projections onto PC1
        projections = X2c @ pc1
        X_proj = np.outer(projections, pc1)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X2[:,0], y=X2[:,1], mode='markers',
                marker=dict(color='#534AB7', size=6, opacity=0.6), name='Data points'))

            if show_proj:
                for i in range(len(X2c)):
                    orig = X2c[i] + X2.mean(axis=0)
                    proj = X_proj[i] + X2.mean(axis=0)
                    fig.add_shape(type='line', x0=orig[0], y0=orig[1],
                        x1=proj[0], y1=proj[1],
                        line=dict(color='rgba(239,159,39,0.3)', width=1))
                fig.add_trace(go.Scatter(
                    x=(X_proj + X2.mean(axis=0))[:,0],
                    y=(X_proj + X2.mean(axis=0))[:,1],
                    mode='markers', marker=dict(color='#EF9F27', size=5, opacity=0.8),
                    name='Projected onto PC1'))

            # draw PC arrows
            scale = np.sqrt(evals2) * 1.5
            for i, (ev, sc, col_p, lbl) in enumerate(zip(
                    evecs2.T, scale, ['#E24B4A','#1D9E75'], ['PC1','PC2'])):
                fig.add_shape(type='line',
                    x0=X2.mean(0)[0]-ev[0]*sc, y0=X2.mean(0)[1]-ev[1]*sc,
                    x1=X2.mean(0)[0]+ev[0]*sc, y1=X2.mean(0)[1]+ev[1]*sc,
                    line=dict(color=col_p, width=3))
                fig.add_annotation(
                    x=X2.mean(0)[0]+ev[0]*sc*1.15,
                    y=X2.mean(0)[1]+ev[1]*sc*1.15,
                    text=f"<b>{lbl}</b> ({var_exp[i]:.1%})",
                    font=dict(color=col_p, size=12), showarrow=False)

            lim = max(abs(X2).max(), abs(X_proj + X2.mean(0)).max()) + 0.5
            fig.update_layout(xaxis=dict(range=[-lim,lim], zeroline=True),
                yaxis=dict(range=[-lim,lim], zeroline=True, scaleanchor='x', scaleratio=1),
                height=430, plot_bgcolor='white',
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("PC1 variance explained", f"{var_exp[0]:.1%}")
        c2.metric("PC2 variance explained", f"{var_exp[1]:.1%}")
        c3.metric("PC1 direction", f"[{pc1[0]:.3f}, {pc1[1]:.3f}]")
        st.info("Orange lines show the reconstruction error — information lost by projecting onto PC1 only.")

    with tab2:
        st.markdown("PCA on a higher-dimensional dataset, projected to 2D for visualisation.")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_classes = st.slider("Classes", 2, 4, 3, key="pca_cls")
            n_feats   = st.slider("Original features (dimensions)", 3, 10, 5, key="pca_f")
            n_samp    = st.slider("Samples per class", 20, 80, 40, key="pca_s")
            seed_hd   = st.slider("Seed", 0, 10, 1, key="pca_seed2")

        np.random.seed(seed_hd)
        centers = np.random.randn(n_classes, n_feats) * 2
        X_hd = np.vstack([np.random.randn(n_samp, n_feats) + centers[c]
                          for c in range(n_classes)])
        y_hd = np.repeat(np.arange(n_classes), n_samp)

        # PCA via SVD
        X_hdc = X_hd - X_hd.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_hdc, full_matrices=False)
        var_ratio = S**2 / (S**2).sum()
        X_2d = X_hdc @ Vt[:2].T

        with col2:
            cls_colors = ['#534AB7','#E24B4A','#1D9E75','#EF9F27']
            fig = go.Figure()
            for c in range(n_classes):
                mask = y_hd == c
                fig.add_trace(go.Scatter(x=X_2d[mask,0], y=X_2d[mask,1],
                    mode='markers', name=f'Class {c+1}',
                    marker=dict(color=cls_colors[c], size=7, opacity=0.75)))
            fig.update_layout(xaxis_title=f"PC1 ({var_ratio[0]:.1%})",
                yaxis_title=f"PC2 ({var_ratio[1]:.1%})",
                height=390, legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"Original: **{n_feats}D** → PCA projection: **2D** "
                    f"(retaining {var_ratio[0]+var_ratio[1]:.1%} of variance)")

    with tab3:
        st.markdown("### Scree plot — how many components to keep?")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_feat_s = st.slider("Number of features", 3, 15, 8, key="pca_sc_f")
            n_samp_s = st.slider("Samples", 50, 300, 120, key="pca_sc_s")
            seed_sc  = st.slider("Seed", 0, 10, 0, key="pca_sc_seed")
            threshold = st.slider("Variance threshold", 0.7, 0.99, 0.90, step=0.01)

        np.random.seed(seed_sc)
        # make data with varying signal strength across dims
        true_dims = max(2, n_feat_s // 2)
        W = np.random.randn(n_feat_s, true_dims)
        Z = np.random.randn(n_samp_s, true_dims)
        X_sc = Z @ W.T + np.random.randn(n_samp_s, n_feat_s) * 0.5
        X_sc -= X_sc.mean(0)
        _, S_sc, _ = np.linalg.svd(X_sc, full_matrices=False)
        var_sc = S_sc**2 / (S_sc**2).sum()
        cumvar = np.cumsum(var_sc)
        n_keep = int(np.searchsorted(cumvar, threshold)) + 1

        with col2:
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=["Variance per component (scree)", "Cumulative variance"])
            fig.add_trace(go.Bar(x=list(range(1, n_feat_s+1)), y=var_sc,
                marker_color=['#E24B4A' if i < n_keep else '#AFA9EC'
                              for i in range(n_feat_s)], name='Individual'),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(1, n_feat_s+1)), y=cumvar,
                mode='lines+markers', line=dict(color='#534AB7', width=2.5),
                name='Cumulative'), row=1, col=2)
            fig.add_hline(y=threshold, line_dash='dash', line_color='#EF9F27',
                annotation_text=f"{threshold:.0%}", row=1, col=2)
            fig.add_vline(x=n_keep, line_dash='dot', line_color='#E24B4A',
                annotation_text=f"keep {n_keep}", row=1, col=1)
            fig.update_xaxes(title_text="Component")
            fig.update_yaxes(title_text="Variance explained")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.metric(f"Components needed for {threshold:.0%} variance",
                  f"{n_keep} out of {n_feat_s}",
                  f"{n_feat_s - n_keep} dimensions removed")


# ═══════════════════════════════════════════════════════════════════════════
# DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════
elif section == "decision_tree":
    st.title("🌳 Decision Tree")
    st.markdown("""
    <div class="concept-card">
    A decision tree repeatedly <b>splits the data</b> on the feature and threshold that best
    separates the classes. Each split is chosen to maximise <b>information gain</b>
    (or minimise Gini impurity). The result is a set of human-readable if/else rules.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Decision boundary", "Gini impurity & splits"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            max_depth = st.slider("Max tree depth", 1, 6, 3)
            n_dt = st.slider("Samples per class", 30, 100, 50, key="dt_n")
            sep_dt = st.slider("Class separation", 0.5, 3.0, 1.5, step=0.1, key="dt_sep")
            seed_dt = st.slider("Seed", 0, 20, 5, key="dt_seed")
            n_classes_dt = st.radio("Classes", [2, 3], horizontal=True)

        from sklearn.tree import DecisionTreeClassifier
        np.random.seed(seed_dt)
        if n_classes_dt == 2:
            X_dt = np.vstack([np.random.randn(n_dt,2)+[-sep_dt/2,0],
                              np.random.randn(n_dt,2)+[sep_dt/2,0]])
            y_dt = np.array([0]*n_dt+[1]*n_dt)
        else:
            X_dt = np.vstack([np.random.randn(n_dt,2)+[-sep_dt,0],
                              np.random.randn(n_dt,2)+[sep_dt,0],
                              np.random.randn(n_dt,2)+[0,sep_dt]])
            y_dt = np.array([0]*n_dt+[1]*n_dt+[2]*n_dt)

        clf_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf_dt.fit(X_dt, y_dt)
        train_acc = clf_dt.score(X_dt, y_dt)

        xx_dt, yy_dt = np.meshgrid(np.linspace(-5,5,150), np.linspace(-5,5,150))
        Z_dt = clf_dt.predict(np.c_[xx_dt.ravel(), yy_dt.ravel()]).reshape(xx_dt.shape)

        dt_colors = ['#534AB7','#E24B4A','#1D9E75']
        bg_colors  = ['rgba(83,74,183,0.12)','rgba(226,75,74,0.12)','rgba(29,158,117,0.12)']

        with col2:
            fig = go.Figure()
            for cls in range(n_classes_dt):
                mask_bg = Z_dt == cls
                fig.add_trace(go.Scatter(
                    x=xx_dt.ravel()[mask_bg.ravel()],
                    y=yy_dt.ravel()[mask_bg.ravel()],
                    mode='markers', marker=dict(color=dt_colors[cls], size=3, opacity=0.12),
                    showlegend=False))
            for cls in range(n_classes_dt):
                mask = y_dt == cls
                fig.add_trace(go.Scatter(x=X_dt[mask,0], y=X_dt[mask,1],
                    mode='markers', name=f'Class {cls}',
                    marker=dict(color=dt_colors[cls], size=7, opacity=0.85,
                        line=dict(color='white', width=0.5))))
            fig.update_layout(xaxis=dict(range=[-5,5]), yaxis=dict(range=[-5,5]),
                height=420, legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Max depth", max_depth)
        c2.metric("Train accuracy", f"{train_acc:.1%}")
        c3.metric("Leaf nodes", clf_dt.get_n_leaves())

        if max_depth >= 5:
            st.warning("Deep tree — likely overfitting. The jagged boundaries memorise training noise.")
        elif max_depth <= 2:
            st.info("Shallow tree — may underfit complex patterns, but generalises well.")
        else:
            st.success("Moderate depth — good balance of expressiveness and generalisability.")

    with tab2:
        st.markdown("### Gini impurity — the split criterion")
        st.markdown('<div class="formula-box">Gini = 1 − Σ pᵢ²</div>', unsafe_allow_html=True)
        st.markdown("Gini = 0 means a pure node (all one class). Gini = 0.5 means perfectly mixed (2 classes).")

        col1, col2 = st.columns([1, 2])
        with col1:
            p_cls1 = st.slider("Fraction of class A in node", 0.0, 1.0, 0.5, step=0.01)
            p_cls2 = 1.0 - p_cls1
            gini = 1 - (p_cls1**2 + p_cls2**2)
            entropy = -(p_cls1*np.log2(p_cls1+1e-9) + p_cls2*np.log2(p_cls2+1e-9))
            st.metric("Gini impurity", f"{gini:.4f}")
            st.metric("Entropy", f"{entropy:.4f}")
            if gini < 0.1:   st.success("Very pure node — good split!")
            elif gini < 0.3: st.info("Moderately pure")
            else:             st.warning("Impure node — tree will try to split further")

        with col2:
            p_range = np.linspace(0.001, 0.999, 300)
            gini_curve    = 1 - (p_range**2 + (1-p_range)**2)
            entropy_curve = -(p_range*np.log2(p_range) + (1-p_range)*np.log2(1-p_range))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_range, y=gini_curve, name='Gini',
                line=dict(color='#534AB7', width=2.5)))
            fig.add_trace(go.Scatter(x=p_range, y=entropy_curve/2, name='Entropy/2',
                line=dict(color='#E24B4A', width=2.5)))
            fig.add_vline(x=p_cls1, line_dash='dash', line_color='#EF9F27',
                annotation_text=f"p={p_cls1:.2f}")
            fig.update_layout(xaxis_title="Fraction of class A",
                yaxis_title="Impurity", height=320,
                legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Information gain — how a split is chosen")
        st.markdown("""
        At each node the tree tries every feature and every threshold.
        It picks the one that maximises:
        """)
        st.markdown('<div class="formula-box">IG = Gini(parent) − [n_L/n · Gini(left) + n_R/n · Gini(right)]</div>',
            unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            n_left  = st.slider("Left child samples", 5, 50, 20)
            p_left  = st.slider("Fraction class A — left",  0.0, 1.0, 0.9, step=0.01)
            n_right = st.slider("Right child samples", 5, 50, 30)
            p_right = st.slider("Fraction class A — right", 0.0, 1.0, 0.1, step=0.01)
        n_total = n_left + n_right
        p_parent = (n_left*p_left + n_right*p_right) / n_total
        gini_parent = 1-(p_parent**2+(1-p_parent)**2)
        gini_left   = 1-(p_left**2+(1-p_left)**2)
        gini_right  = 1-(p_right**2+(1-p_right)**2)
        ig = gini_parent - (n_left/n_total*gini_left + n_right/n_total*gini_right)
        with col2:
            st.code(f"""
Parent gini  = {gini_parent:.4f}  (p_A = {p_parent:.2f})
Left gini    = {gini_left:.4f}   (p_A = {p_left:.2f}, n={n_left})
Right gini   = {gini_right:.4f}  (p_A = {p_right:.2f}, n={n_right})

Information gain = {gini_parent:.4f} − ({n_left}/{n_total}·{gini_left:.4f} + {n_right}/{n_total}·{gini_right:.4f})
                 = {ig:.4f}
""", language="text")
            if ig > 0.15: st.success("Excellent split — large reduction in impurity")
            elif ig > 0.05: st.info("Decent split")
            else: st.warning("Poor split — little gain")


# ═══════════════════════════════════════════════════════════════════════════
# NAIVE BAYES
# ═══════════════════════════════════════════════════════════════════════════
elif section == "naive_bayes":
    st.title("📊 Naive Bayes")
    st.markdown("""
    <div class="concept-card">
    Naive Bayes applies <b>Bayes' theorem</b> with the "naive" assumption that features are
    independent given the class. Despite its simplicity it works surprisingly well for
    text classification, spam filtering and medical diagnosis.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">P(class | features) ∝ P(class) · Π P(featureᵢ | class)</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Bayes theorem live", "Gaussian Naive Bayes classifier"])

    with tab1:
        st.markdown("### Update your belief with evidence")
        col1, col2 = st.columns([1, 2])
        with col1:
            prior = st.slider("Prior P(disease)", 0.001, 0.5, 0.01, step=0.001,
                format="%.3f", help="How common is the disease in the population?")
            sensitivity = st.slider("Sensitivity P(+|disease)", 0.5, 1.0, 0.95, step=0.01,
                help="True positive rate of the test")
            specificity = st.slider("Specificity P(−|healthy)", 0.5, 1.0, 0.90, step=0.01,
                help="True negative rate of the test")
            test_result = st.radio("Test result", ["Positive ✅", "Negative ❌"])

        fp_rate = 1 - specificity
        p_pos   = prior*sensitivity + (1-prior)*fp_rate
        p_neg   = prior*(1-sensitivity) + (1-prior)*specificity

        if test_result == "Positive ✅":
            posterior = prior * sensitivity / (p_pos + 1e-12)
            evidence_str = "positive"
        else:
            posterior = prior * (1-sensitivity) / (p_neg + 1e-12)
            evidence_str = "negative"

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Prior P(disease)', f'Posterior P(disease|test {evidence_str})'],
                y=[prior, posterior],
                marker_color=['#AFA9EC','#534AB7' if posterior > prior else '#E24B4A'],
                text=[f"{prior:.4f}", f"{posterior:.4f}"],
                textposition='outside'))
            fig.update_layout(yaxis=dict(range=[0, min(1, max(posterior, prior)*1.5)]),
                height=320, showlegend=False, yaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Prior", f"{prior:.4f}")
        c2.metric("Posterior", f"{posterior:.4f}")
        change = (posterior - prior) / (prior + 1e-12)
        c3.metric("Belief change", f"{change:+.1f}×")
        st.markdown(f"""
        **Bayes formula applied:**
        - P(disease | positive) = P(+|disease)·P(disease) / P(+) = {sensitivity:.2f}·{prior:.3f} / {p_pos:.4f} = **{posterior:.4f}**
        """)
        if prior < 0.01 and posterior < 0.3:
            st.info("Even with a positive test, the posterior is low because the disease is rare — this is the **base rate fallacy**.")

    with tab2:
        st.markdown("### Gaussian Naive Bayes — 2D feature classifier")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_nb = st.slider("Samples per class", 20, 80, 40, key="nb_n")
            sep_nb = st.slider("Class separation", 0.5, 3.0, 1.5, step=0.1, key="nb_sep")
            seed_nb = st.slider("Seed", 0, 10, 1, key="nb_seed")
            test_x1 = st.slider("Test point — feature 1", -4.0, 4.0, 0.5, step=0.1)
            test_x2 = st.slider("Test point — feature 2", -4.0, 4.0, 0.5, step=0.1)

        np.random.seed(seed_nb)
        X_nb0 = np.random.randn(n_nb, 2) + np.array([-sep_nb/2, 0])
        X_nb1 = np.random.randn(n_nb, 2) + np.array([sep_nb/2,  0])
        X_nb  = np.vstack([X_nb0, X_nb1])
        y_nb  = np.array([0]*n_nb + [1]*n_nb)

        # fit Gaussian NB manually
        means = np.array([X_nb[y_nb==c].mean(0) for c in [0,1]])
        stds  = np.array([X_nb[y_nb==c].std(0)  for c in [0,1]])
        priors_nb = np.array([0.5, 0.5])

        def gauss_pdf(x, mu, sigma):
            return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi) + 1e-9)

        def nb_predict_proba(x):
            log_probs = []
            for c in [0,1]:
                lp = np.log(priors_nb[c])
                lp += np.sum(np.log(gauss_pdf(x, means[c], stds[c]) + 1e-12))
                log_probs.append(lp)
            log_probs = np.array(log_probs)
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            return probs / probs.sum()

        # decision boundary grid
        xx_nb, yy_nb = np.meshgrid(np.linspace(-5,5,120), np.linspace(-5,5,120))
        grid_nb = np.c_[xx_nb.ravel(), yy_nb.ravel()]
        Z_nb = np.array([nb_predict_proba(pt)[1] for pt in grid_nb]).reshape(xx_nb.shape)

        test_proba = nb_predict_proba(np.array([test_x1, test_x2]))

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Contour(x=np.linspace(-5,5,120), y=np.linspace(-5,5,120),
                z=Z_nb, colorscale=[[0,'rgba(83,74,183,0.2)'],[0.5,'white'],[1,'rgba(226,75,74,0.2)']],
                showscale=False, contours=dict(showlabels=False)))
            fig.add_trace(go.Scatter(x=X_nb0[:,0], y=X_nb0[:,1], mode='markers',
                name='Class 0', marker=dict(color='#534AB7', size=7, opacity=0.75)))
            fig.add_trace(go.Scatter(x=X_nb1[:,0], y=X_nb1[:,1], mode='markers',
                name='Class 1', marker=dict(color='#E24B4A', size=7, opacity=0.75)))
            pred_cls = int(test_proba[1] >= 0.5)
            fig.add_trace(go.Scatter(x=[test_x1], y=[test_x2], mode='markers',
                name=f'Test → Class {pred_cls}',
                marker=dict(color=['#534AB7','#E24B4A'][pred_cls], size=14,
                    symbol='star', line=dict(color='black',width=1.5))))
            fig.update_layout(xaxis=dict(range=[-5,5]), yaxis=dict(range=[-5,5]),
                height=400, legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("P(Class 0 | x)", f"{test_proba[0]:.4f}")
        c2.metric("P(Class 1 | x)", f"{test_proba[1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# CNN — CONVOLUTIONAL LAYER
# ═══════════════════════════════════════════════════════════════════════════
elif section == "cnn":
    st.title("🖼️ Convolutional Layer (CNN)")
    st.markdown("""
    <div class="concept-card">
    A convolutional layer slides a small <b>kernel</b> (filter) across an input,
    computing a dot product at every position. This detects local patterns —
    edges, textures, shapes — regardless of where they appear in the image.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Kernel convolution visualised", "Common kernels & effects"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            grid_size = st.slider("Input size (N×N)", 6, 12, 8)
            input_type = st.selectbox("Input pattern",
                ["Random", "Vertical edge", "Horizontal edge", "Diagonal", "Checkerboard"])
            kernel_preset = st.selectbox("Kernel (3×3)",
                ["Edge detect (vertical)", "Edge detect (horizontal)",
                 "Sharpen", "Blur (average)", "Custom"])
            stride = st.radio("Stride", [1, 2], horizontal=True)
            padding = st.checkbox("Zero padding (same size output)", value=False)

        np.random.seed(42)
        G = grid_size
        if input_type == "Random":
            inp = np.random.rand(G, G)
        elif input_type == "Vertical edge":
            inp = np.zeros((G,G)); inp[:, G//2:] = 1.0
        elif input_type == "Horizontal edge":
            inp = np.zeros((G,G)); inp[G//2:, :] = 1.0
        elif input_type == "Diagonal":
            inp = np.array([[1.0 if abs(i-j)<2 else 0.0 for j in range(G)] for i in range(G)])
        else:
            inp = np.array([[(i+j)%2*1.0 for j in range(G)] for i in range(G)])

        kernels = {
            "Edge detect (vertical)":   np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
            "Edge detect (horizontal)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
            "Sharpen":                  np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
            "Blur (average)":           np.ones((3,3))/9,
            "Custom":                   np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),
        }
        K = kernels[kernel_preset]

        # convolution
        pad = 1 if padding else 0
        inp_padded = np.pad(inp, pad) if padding else inp
        out_size = (inp_padded.shape[0]-3)//stride+1
        feature_map = np.zeros((out_size, out_size))
        for i in range(out_size):
            for j in range(out_size):
                patch = inp_padded[i*stride:i*stride+3, j*stride:j*stride+3]
                feature_map[i,j] = np.sum(patch * K)

        with col2:
            fig = make_subplots(rows=1, cols=3,
                subplot_titles=["Input", "Kernel (3×3)", "Feature map (output)"],
                column_widths=[0.45, 0.15, 0.4])
            fig.add_trace(go.Heatmap(z=inp, colorscale='Greys', showscale=False,
                xgap=1, ygap=1), row=1, col=1)
            fig.add_trace(go.Heatmap(z=K, colorscale='RdBu', showscale=False,
                xgap=2, ygap=2,
                text=[[f"{K[i,j]:.1f}" for j in range(3)] for i in range(3)],
                texttemplate="%{text}", textfont=dict(size=11)), row=1, col=2)
            fig.add_trace(go.Heatmap(z=feature_map, colorscale='RdBu',
                showscale=False, xgap=1, ygap=1), row=1, col=3)
            fig.update_yaxes(autorange='reversed')
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Input size", f"{G}×{G}")
        c2.metric("Output size", f"{out_size}×{out_size}")
        c3.metric("Parameters in kernel", "9 weights + 1 bias")
        st.markdown(f"**Key insight:** the same 9 weights are reused at every position — "
                    f"this *weight sharing* gives CNNs translation invariance and makes them vastly more "
                    f"efficient than a fully-connected layer ({G*G*out_size*out_size:,} weights vs 9).")

    with tab2:
        st.markdown("### How different kernels highlight different features")
        np.random.seed(7)
        # make a structured test image
        test_img = np.zeros((16,16))
        test_img[3:13, 3:13] = 0.5
        test_img[5:11, 5:11] = 1.0
        test_img[2, :] = 0.8; test_img[:, 2] = 0.8

        kernel_gallery = {
            "Original":               np.array([[0,0,0],[0,1,0],[0,0,0]]),
            "Vertical edges":         np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
            "Horizontal edges":       np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
            "Blur":                   np.ones((3,3))/9,
            "Sharpen":                np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        }

        results = {}
        for name, k in kernel_gallery.items():
            fm = np.zeros((14,14))
            for i in range(14):
                for j in range(14):
                    fm[i,j] = np.sum(test_img[i:i+3,j:j+3]*k)
            results[name] = fm

        fig = make_subplots(rows=1, cols=5,
            subplot_titles=list(kernel_gallery.keys()))
        for idx, (name, fm) in enumerate(results.items()):
            fig.add_trace(go.Heatmap(z=fm, colorscale='RdBu' if idx>0 else 'Greys',
                showscale=False, xgap=1, ygap=1), row=1, col=idx+1)
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Same input image, 5 different 3×3 kernels. CNNs learn these kernels automatically from data.")


# ═══════════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM
# ═══════════════════════════════════════════════════════════════════════════
elif section == "attention":
    st.title("👁️ Attention Mechanism")
    st.markdown("""
    <div class="concept-card">
    Attention allows a model to <b>selectively focus</b> on different parts of the input
    when producing each output. In transformers, every token computes a <b>Query</b>,
    receives <b>Keys</b> from all tokens, and uses the dot product similarity to decide
    how much to <b>attend</b> to each position.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Attention weights visualised", "Softmax & temperature"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            sentence = st.text_input("Input sentence (space-separated tokens)",
                "the cat sat on the mat")
            tokens = sentence.strip().split()
            n_tok = len(tokens)
            query_idx = st.slider("Query token (which word is 'looking')",
                0, max(n_tok-1, 0), min(2, n_tok-1))
            d_k = st.select_slider("Key dimension dₖ", [4, 8, 16, 32], value=8)
            seed_att = st.slider("Seed (random QK weights)", 0, 20, 5, key="att_seed")
            temperature = st.slider("Temperature (1/√dₖ scaling)", 0.1, 2.0, 1.0, step=0.1)

        np.random.seed(seed_att)
        # random embeddings for demo
        embeddings = np.random.randn(n_tok, d_k)
        W_Q = np.random.randn(d_k, d_k) * 0.5
        W_K = np.random.randn(d_k, d_k) * 0.5
        W_V = np.random.randn(d_k, d_k) * 0.5

        Q = embeddings @ W_Q
        K = embeddings @ W_K
        V = embeddings @ W_V

        # raw scores for the query token
        scores = Q[query_idx] @ K.T / (np.sqrt(d_k) * temperature)
        attn_weights = np.exp(scores - scores.max())
        attn_weights /= attn_weights.sum()

        # full attention matrix
        all_scores = Q @ K.T / (np.sqrt(d_k) * temperature)
        all_weights = np.exp(all_scores - all_scores.max(axis=1, keepdims=True))
        all_weights /= all_weights.sum(axis=1, keepdims=True)

        with col2:
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=[f'Attention from "{tokens[query_idx]}"',
                                "Full attention matrix"])
            fig.add_trace(go.Bar(x=tokens, y=attn_weights,
                marker_color=['#E24B4A' if i==query_idx else '#534AB7' for i in range(n_tok)],
                text=[f"{w:.3f}" for w in attn_weights], textposition='outside',
                showlegend=False), row=1, col=1)
            fig.add_trace(go.Heatmap(z=all_weights, x=tokens, y=tokens,
                colorscale='Blues', showscale=False,
                text=[[f"{all_weights[i,j]:.2f}" for j in range(n_tok)] for i in range(n_tok)],
                texttemplate="%{text}", textfont=dict(size=10)), row=1, col=2)
            fig.update_yaxes(autorange='reversed', row=1, col=2)
            fig.update_xaxes(title_text="Key token", row=1, col=1)
            fig.update_yaxes(title_text="Attention weight", row=1, col=1)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        top_token = tokens[np.argmax(attn_weights)]
        st.markdown(f'**"{tokens[query_idx]}"** attends most strongly to **"{top_token}"** '
                    f'(weight = {attn_weights.max():.3f})')

    with tab2:
        st.markdown("### Softmax converts raw scores into attention weights")
        st.markdown('<div class="formula-box">softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)</div>',
            unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            n_scores = st.slider("Number of keys", 3, 8, 5)
            raw_scores = [st.slider(f"Score z{i+1}", -3.0, 3.0,
                [1.5, -0.5, 2.0, 0.3, -1.0, 0.8, -0.2, 1.1][i], step=0.1, key=f"sm_{i}")
                for i in range(n_scores)]
            temp_sm = st.slider("Temperature τ", 0.1, 3.0, 1.0, step=0.1,
                help="Low T → sharp (winner-take-all). High T → uniform.")

        scores_arr = np.array(raw_scores) / temp_sm
        sm = np.exp(scores_arr - scores_arr.max())
        sm /= sm.sum()

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"k{i+1}" for i in range(n_scores)],
                y=raw_scores, name='Raw scores', marker_color='#AFA9EC', opacity=0.7))
            fig.add_trace(go.Bar(x=[f"k{i+1}" for i in range(n_scores)],
                y=sm, name='After softmax', marker_color='#534AB7'))
            fig.update_layout(barmode='group', height=320,
                legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Temperature τ={temp_sm}:** {'sharp focus on top key' if temp_sm < 0.5 else 'uniform attention across all keys' if temp_sm > 2 else 'balanced attention'}")
        st.caption("In transformers τ = 1/√dₖ. Low temperature → the model is more decisive; high → more exploratory.")


# ═══════════════════════════════════════════════════════════════════════════
# PROBABILITY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "probability":
    st.title("🎲 Probability Distributions")
    st.markdown("""
    <div class="concept-card">
    A probability distribution describes how likely each value of a random variable is.
    Distributions are everywhere in ML: loss functions assume distributions over errors,
    variational autoencoders sample from Gaussians, and Bayesian methods put distributions
    over parameters.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Normal (Gaussian)", "Binomial & Poisson", "Comparing distributions"])

    with tab1:
        st.markdown('<div class="formula-box">f(x) = (1/σ√2π) · exp(−(x−μ)²/2σ²)</div>',
            unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            mu    = st.slider("Mean μ", -4.0, 4.0, 0.0, step=0.1, key="nd_mu")
            sigma = st.slider("Std dev σ", 0.1, 3.0, 1.0, step=0.1, key="nd_sigma")
            show_68 = st.checkbox("Show 68–95–99.7 rule", value=True)
            x_query = st.slider("Query point x", -6.0, 6.0, 1.0, step=0.1)

        x_nd = np.linspace(mu-5*sigma, mu+5*sigma, 400)
        y_nd = np.exp(-0.5*((x_nd-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
        pdf_at_x = np.exp(-0.5*((x_query-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
        cdf_at_x = float(0.5*(1+np.sign(x_query-mu)*
                    (1-np.exp(-0.7071*(abs(x_query-mu)/sigma)**1.6)))) # approx

        with col2:
            fig = go.Figure()
            if show_68:
                for k, alpha, label in [(1,0.20,'68%'),(2,0.12,'95%'),(3,0.07,'99.7%')]:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_nd[(x_nd>=mu-k*sigma)&(x_nd<=mu+k*sigma)],
                                          x_nd[(x_nd>=mu-k*sigma)&(x_nd<=mu+k*sigma)][::-1]]),
                        y=np.concatenate([y_nd[(x_nd>=mu-k*sigma)&(x_nd<=mu+k*sigma)],
                                          np.zeros(((x_nd>=mu-k*sigma)&(x_nd<=mu+k*sigma)).sum())]),
                        fill='toself', fillcolor=f'rgba(83,74,183,{alpha})',
                        line=dict(color='rgba(0,0,0,0)'), name=f'±{k}σ ({label})'))
            fig.add_trace(go.Scatter(x=x_nd, y=y_nd, name='PDF',
                line=dict(color='#534AB7', width=2.5)))
            fig.add_vline(x=mu, line_dash='dot', line_color='#EF9F27',
                annotation_text=f"μ={mu}")
            fig.add_vline(x=x_query, line_dash='dash', line_color='#E24B4A',
                annotation_text=f"x={x_query:.1f}")
            fig.update_layout(xaxis_title="x", yaxis_title="Probability density",
                height=380, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("PDF f(x)", f"{pdf_at_x:.4f}")
        c2.metric("Mean μ", mu)
        c3.metric("Variance σ²", f"{sigma**2:.2f}")

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            dist_type = st.radio("Distribution", ["Binomial", "Poisson"])
            if dist_type == "Binomial":
                n_bin = st.slider("n (trials)", 1, 50, 20)
                p_bin = st.slider("p (success prob)", 0.01, 0.99, 0.3, step=0.01)
                k_vals = np.arange(0, n_bin+1)
                from math import comb
                pmf = np.array([comb(n_bin,k)*(p_bin**k)*((1-p_bin)**(n_bin-k)) for k in k_vals])
                mean_d, var_d = n_bin*p_bin, n_bin*p_bin*(1-p_bin)
                title = f"Binomial(n={n_bin}, p={p_bin})"
            else:
                lam = st.slider("λ (rate)", 0.5, 15.0, 4.0, step=0.5)
                k_vals = np.arange(0, int(lam*3)+1)
                pmf = np.exp(-lam) * lam**k_vals / np.array([np.math.factorial(k) for k in k_vals])
                mean_d, var_d = lam, lam
                title = f"Poisson(λ={lam})"

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=k_vals, y=pmf, name='PMF',
                marker_color='#534AB7', opacity=0.8))
            fig.add_vline(x=mean_d, line_dash='dash', line_color='#E24B4A',
                annotation_text=f"mean={mean_d:.2f}")
            fig.update_layout(xaxis_title="k", yaxis_title="P(X=k)",
                title=title, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        c1.metric("Mean", f"{mean_d:.3f}")
        c2.metric("Variance", f"{var_d:.3f}")

    with tab3:
        st.markdown("### Shape comparison — same mean, different distributions")
        mean_cmp = st.slider("Shared mean", -2.0, 2.0, 0.0, step=0.2)
        x_cmp = np.linspace(-6, 8, 400)
        fig = go.Figure()
        for sigma_cmp, col_c, lbl in [(0.5,'#E24B4A','σ=0.5 (narrow)'),(1.0,'#534AB7','σ=1.0'),(2.0,'#1D9E75','σ=2.0 (wide)')]:
            y_cmp = np.exp(-0.5*((x_cmp-mean_cmp)/sigma_cmp)**2)/(sigma_cmp*np.sqrt(2*np.pi))
            fig.add_trace(go.Scatter(x=x_cmp, y=y_cmp, name=lbl, line=dict(color=col_c, width=2.5)))
        fig.update_layout(xaxis_title="x", yaxis_title="PDF", height=320,
            legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        - **Narrow (low σ)** — model is very certain, low entropy
        - **Wide (high σ)** — high uncertainty, high entropy
        - In VAEs and Bayesian ML, learning the right σ is as important as learning the mean
        """)


# ═══════════════════════════════════════════════════════════════════════════
# SVD — MATRIX DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "svd":
    st.title("✂️ SVD — Singular Value Decomposition")
    st.markdown("""
    <div class="concept-card">
    SVD decomposes any matrix into three simpler matrices: <b>A = U Σ Vᵀ</b>.
    It reveals the intrinsic structure of data. Key applications: PCA, image compression,
    recommendation systems, and understanding what a neural network layer has learned.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">A = U · Σ · Vᵀ &nbsp;&nbsp;&nbsp; (m×n) = (m×m)(m×n)(n×n)</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Image compression", "Low-rank approximation explorer"])

    with tab1:
        st.markdown("### SVD as lossy compression — keep only the top k singular values")
        col1, col2 = st.columns([1, 2])
        with col1:
            img_type = st.selectbox("Test image", ["Gradient", "Stripes", "Checkerboard", "Random noise", "Smooth blob"])
            k_svd = st.slider("Rank k (singular values kept)", 1, 30, 5)
            img_size = 40

        np.random.seed(0)
        if img_type == "Gradient":
            img = np.outer(np.linspace(0,1,img_size), np.linspace(0,1,img_size))
        elif img_type == "Stripes":
            img = np.tile(np.linspace(0,1,img_size), (img_size,1))
            img[::2] = 1 - img[::2]
        elif img_type == "Checkerboard":
            img = np.array([[(i+j)%2 for j in range(img_size)] for i in range(img_size)], dtype=float)
        elif img_type == "Random noise":
            img = np.random.rand(img_size, img_size)
        else:
            cx, cy = img_size//2, img_size//2
            img = np.array([[np.exp(-((i-cx)**2+(j-cy)**2)/80) for j in range(img_size)] for i in range(img_size)])

        U, S, Vt = np.linalg.svd(img)
        k_capped = min(k_svd, len(S))
        img_approx = (U[:, :k_capped] * S[:k_capped]) @ Vt[:k_capped, :]
        error = np.mean((img - img_approx)**2)
        var_captured = (S[:k_capped]**2).sum() / (S**2).sum()
        compression = k_capped*(img_size+img_size+1) / (img_size*img_size)

        with col2:
            fig = make_subplots(rows=1, cols=3,
                subplot_titles=["Original", f"Rank-{k_capped} approx", "Singular values"])
            fig.add_trace(go.Heatmap(z=img, colorscale='Greys', showscale=False), row=1, col=1)
            fig.add_trace(go.Heatmap(z=img_approx, colorscale='Greys', showscale=False), row=1, col=2)
            fig.add_trace(go.Bar(x=list(range(1,len(S)+1)), y=S,
                marker_color=['#534AB7' if i<k_capped else '#AFA9EC' for i in range(len(S))],
                showlegend=False), row=1, col=3)
            fig.add_vline(x=k_capped+0.5, line_dash='dash', line_color='#E24B4A', row=1, col=3)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rank k", k_capped)
        c2.metric("Variance captured", f"{var_captured:.1%}")
        c3.metric("Reconstruction MSE", f"{error:.4f}")
        c4.metric("Storage ratio", f"{compression:.2f}×", help="<1 means compressed")
        st.caption(f"Blue bars = kept singular values. Grey = discarded. "
                   f"Original needs {img_size}² = {img_size**2} values; "
                   f"rank-{k_capped} needs {k_capped*(2*img_size+1)} values.")

    with tab2:
        st.markdown("### The three matrices U, Σ, Vᵀ explained")
        col1, col2 = st.columns([1, 2])
        with col1:
            m_svd = st.slider("Rows m", 3, 8, 4, key="svd_m")
            n_svd = st.slider("Cols n", 3, 8, 5, key="svd_n")
            rank_svd = st.slider("True rank of matrix", 1, min(m_svd,n_svd), 2)
            seed_svd = st.slider("Seed", 0, 10, 0, key="svd_seed")

        np.random.seed(seed_svd)
        # build a low-rank matrix
        A_true = np.random.randn(m_svd, rank_svd) @ np.random.randn(rank_svd, n_svd)
        A_noisy = A_true + np.random.randn(m_svd, n_svd) * 0.5
        U_s, S_s, Vt_s = np.linalg.svd(A_noisy, full_matrices=False)

        with col2:
            fig = make_subplots(rows=1, cols=4,
                subplot_titles=["A (noisy)", "U (left singular)", "Σ (diagonal)", "Vᵀ (right singular)"],
                column_widths=[0.3, 0.25, 0.15, 0.3])
            fig.add_trace(go.Heatmap(z=A_noisy, colorscale='RdBu', showscale=False,
                zmid=0), row=1, col=1)
            fig.add_trace(go.Heatmap(z=U_s, colorscale='RdBu', showscale=False,
                zmid=0), row=1, col=2)
            fig.add_trace(go.Heatmap(z=np.diag(S_s), colorscale='Blues',
                showscale=False), row=1, col=3)
            fig.add_trace(go.Heatmap(z=Vt_s, colorscale='RdBu', showscale=False,
                zmid=0), row=1, col=4)
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**U** — how original rows relate to latent dimensions  \n"
                    "**Σ** — importance of each latent dimension (singular values)  \n"
                    "**Vᵀ** — how original columns relate to latent dimensions")
        import pandas as pd
        st.dataframe(pd.DataFrame({
            "Singular value": [f"σ{i+1} = {S_s[i]:.3f}" for i in range(len(S_s))],
            "Variance share": [f"{S_s[i]**2/(S_s**2).sum():.1%}" for i in range(len(S_s))],
            "Cumulative": [f"{(S_s[:i+1]**2).sum()/(S_s**2).sum():.1%}" for i in range(len(S_s))],
        }), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# REACT LOOP
# ═══════════════════════════════════════════════════════════════════════════
elif section == "react_loop":
    st.title("🔄 ReAct Loop")
    st.markdown("""
    <div class="concept-card">
    <b>ReAct</b> (Reason + Act) is the core loop of a tool-using agent.
    At each step the model: <b>Reasons</b> about what to do next,
    <b>Acts</b> by calling a tool, then <b>Observes</b> the result —
    repeating until the task is complete.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Step-by-step simulation", "Loop anatomy"])

    with tab1:
        st.markdown("### Simulate an agent solving a task")

        SCENARIOS = {
            "Find the weather and plan an outfit": [
                ("Thought", "I need to check the current weather to recommend an outfit.",
                 "🧠", "#534AB7"),
                ("Action",  'get_weather(location="Sofia, Bulgaria")',
                 "🔧", "#EF9F27"),
                ("Observation", '{"temp": 12, "condition": "rainy", "wind_mph": 15}',
                 "👁️", "#1D9E75"),
                ("Thought", "It is cold and rainy. I should recommend a waterproof jacket and layers.",
                 "🧠", "#534AB7"),
                ("Action",  'search_web(query="best rainy day outfits 12 degrees")',
                 "🔧", "#EF9F27"),
                ("Observation", "Results: waterproof jacket, warm underlayer, waterproof shoes recommended.",
                 "👁️", "#1D9E75"),
                ("Thought", "I have enough information to give a complete recommendation.",
                 "🧠", "#534AB7"),
                ("Final Answer", "Wear a waterproof jacket over a warm underlayer, waterproof shoes. Bring an umbrella.",
                 "✅", "#E24B4A"),
            ],
            "Calculate compound interest and summarise": [
                ("Thought", "I need to calculate compound interest. I should use the calculator tool.",
                 "🧠", "#534AB7"),
                ("Action",  'calculator(expression="1000 * (1 + 0.05/12) ** (12*3)")',
                 "🔧", "#EF9F27"),
                ("Observation", '{"result": 1161.62}',
                 "👁️", "#1D9E75"),
                ("Thought", "The result is $1161.62. Now I should look up the current savings rate to compare.",
                 "🧠", "#534AB7"),
                ("Action",  'search_web(query="average savings account interest rate 2025")',
                 "🔧", "#EF9F27"),
                ("Observation", "Average savings rate is around 4.5% APY as of 2025.",
                 "👁️", "#1D9E75"),
                ("Thought", "I can now provide a complete answer comparing both options.",
                 "🧠", "#534AB7"),
                ("Final Answer", "$1000 grows to $1161.62 at 5% over 3 years. This beats the average 4.5% APY savings rate.",
                 "✅", "#E24B4A"),
            ],
            "Research and write a short report": [
                ("Thought", "I need to gather information before writing. Let me search for recent data.",
                 "🧠", "#534AB7"),
                ("Action",  'search_web(query="transformer architecture key innovations 2024")',
                 "🔧", "#EF9F27"),
                ("Observation", "Results: mixture-of-experts, flash attention, grouped query attention, long context.",
                 "👁️", "#1D9E75"),
                ("Thought", "Good, I have key points. Let me also check the latest model benchmarks.",
                 "🧠", "#534AB7"),
                ("Action",  'search_web(query="LLM benchmark results 2024 MMLU HumanEval")',
                 "🔧", "#EF9F27"),
                ("Observation", "GPT-4o: MMLU 88.7%, Claude 3.5: MMLU 88.3%, Gemini Ultra: 90.0%",
                 "👁️", "#1D9E75"),
                ("Thought", "I now have enough to write a concise, factual summary report.",
                 "🧠", "#534AB7"),
                ("Final Answer", "Report written: 2024 saw MoE scaling, long-context models, and top MMLU scores reaching 90%.",
                 "✅", "#E24B4A"),
            ],
        }

        col1, col2 = st.columns([1, 2])
        with col1:
            scenario = st.selectbox("Task scenario", list(SCENARIOS.keys()))
            steps = SCENARIOS[scenario]
            n_steps = len(steps)
            step_idx = st.slider("Show up to step", 1, n_steps, 1,
                help="Drag to reveal each step of the agent loop")
            st.markdown("---")
            st.markdown("**Step types:**")
            for label, col_s in [("🧠 Thought","#534AB7"),("🔧 Action","#EF9F27"),
                                   ("👁️ Observation","#1D9E75"),("✅ Final Answer","#E24B4A")]:
                st.markdown(f'<span style="color:{col_s}">●</span> {label}', unsafe_allow_html=True)

        with col2:
            for i, (stype, content_s, icon, color) in enumerate(steps[:step_idx]):
                border = "3px solid" if stype == "Final Answer" else "2px solid"
                bg = "#fef9e7" if stype == "Thought" else \
                     "#fff8f0" if stype == "Action" else \
                     "#f0faf5" if stype == "Observation" else "#fdecea"
                st.markdown(f"""
                <div style="border-left:{border} {color};background:{bg};
                            padding:0.7rem 1rem;border-radius:0 8px 8px 0;
                            margin-bottom:8px">
                    <div style="font-size:0.75rem;font-weight:600;color:{color};
                                text-transform:uppercase;letter-spacing:0.05em">
                        {icon} {stype} — step {i+1}</div>
                    <div style="font-family:monospace;font-size:0.9rem;margin-top:4px">
                        {content_s}</div>
                </div>
                """, unsafe_allow_html=True)

            if step_idx < n_steps:
                remaining = n_steps - step_idx
                st.caption(f"▶ {remaining} more step{'s' if remaining>1 else ''} to go — drag the slider")
            else:
                st.success("✅ Agent completed the task!")

        # metrics
        steps_shown = steps[:step_idx]
        thoughts = sum(1 for s in steps_shown if s[0]=="Thought")
        actions  = sum(1 for s in steps_shown if s[0]=="Action")
        obs      = sum(1 for s in steps_shown if s[0]=="Observation")
        c1,c2,c3 = st.columns(3)
        c1.metric("🧠 Thoughts", thoughts)
        c2.metric("🔧 Tool calls", actions)
        c3.metric("👁️ Observations", obs)

    with tab2:
        st.markdown("### The ReAct loop — anatomy")
        # draw the loop as a flow diagram using plotly
        fig = go.Figure()
        fig.update_layout(xaxis=dict(visible=False, range=[0,10]),
            yaxis=dict(visible=False, range=[0,10]),
            height=420, plot_bgcolor="white",
            margin=dict(l=20,r=20,t=20,b=20))

        nodes = [
            (5, 8.5, "TASK / GOAL", "#E8E6FF", "#534AB7", 14),
            (2, 6,   "🧠 REASON\n(Thought)", "#FFF8E1", "#B8860B", 12),
            (8, 6,   "🔧 ACT\n(Tool call)", "#FFF3E0", "#E65100", 12),
            (5, 3.5, "👁️ OBSERVE\n(Result)", "#E8F5E9", "#1B5E20", 12),
            (5, 1,   "✅ FINAL\nANSWER", "#FFEBEE", "#B71C1C", 12),
        ]
        for x, y, label, fill, border, fsize in nodes:
            w, h = 1.5, 0.8
            fig.add_shape(type="rect", x0=x-w, y0=y-h, x1=x+w, y1=y+h,
                fillcolor=fill, line=dict(color=border, width=2.5), layer="below")
            for di, line in enumerate(label.split("\n")):
                fig.add_annotation(x=x, y=y + 0.2 - di*0.45, text=line,
                    showarrow=False, font=dict(size=fsize, color=border))

        arrows = [
            (5, 7.7, 5, 6.8, "Start"),
            (3.5, 6, 6.5, 6, "decide action"),
            (8, 5.2, 6.5, 4.3, "tool result"),
            (3.5, 4.3, 2, 5.2, "re-reason"),
            (5, 2.7, 5, 1.8, "done?"),
            (3.4, 3.5, 1.5, 3.5, ""),
            (1.5, 3.5, 1.5, 6, ""),
            (1.5, 6, 0.5, 6, "loop back"),
        ]
        for x0,y0,x1,y1,lbl in arrows:
            fig.add_shape(type="line", x0=x0,y0=y0,x1=x1,y1=y1,
                line=dict(color="#888", width=1.5))
            if lbl:
                mx,my = (x0+x1)/2,(y0+y1)/2
                fig.add_annotation(x=mx,y=my,text=lbl,showarrow=False,
                    font=dict(size=9,color="#555"),
                    bgcolor="rgba(255,255,255,0.7)",borderpad=1)

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        The agent loops through **Reason → Act → Observe** until it decides it has enough
        information to produce a final answer. Each iteration gives it new grounding from
        the real world via tools.
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL USE
# ═══════════════════════════════════════════════════════════════════════════
elif section == "tool_use":
    st.title("🔧 Tool Use")
    st.markdown("""
    <div class="concept-card">
    Tool use lets an LLM extend beyond its training data by calling <b>external functions</b> —
    web search, calculators, databases, APIs, code interpreters. The model outputs a structured
    <b>tool call</b>, the result is injected back as an observation, and the model continues.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Tool call anatomy", "Tool selection & routing"])

    with tab1:
        st.markdown("### How a tool call works end to end")

        TOOLS_DEMO = {
            "get_weather": {
                "description": "Get current weather for a location",
                "parameters": {"location": "string — city name or coordinates",
                               "units": "string — 'celsius' or 'fahrenheit'"},
                "example_call": '{"location": "Sofia, Bulgaria", "units": "celsius"}',
                "example_result": '{"temp": 12, "condition": "partly cloudy", "humidity": 65, "wind_kph": 20}',
            },
            "web_search": {
                "description": "Search the web and return top results",
                "parameters": {"query": "string — search query",
                               "max_results": "integer — number of results (1-10)"},
                "example_call": '{"query": "latest transformer architecture research", "max_results": 3}',
                "example_result": '["FlashAttention-3 reduces memory...", "Mamba SSMs challenge...", "Mixture-of-experts scaling..."]',
            },
            "code_interpreter": {
                "description": "Execute Python code and return output",
                "parameters": {"code": "string — valid Python code to execute"},
                "example_call": '{"code": "import numpy as np\\nprint(np.linalg.norm([3, 4]))"}',
                "example_result": '{"stdout": "5.0\\n", "stderr": "", "exit_code": 0}',
            },
            "database_query": {
                "description": "Run a SQL query on the company database",
                "parameters": {"sql": "string — SQL SELECT statement",
                               "database": "string — database name"},
                "example_call": '{"sql": "SELECT AVG(score) FROM users WHERE active=1", "database": "prod"}',
                "example_result": '{"rows": [{"AVG(score)": 87.3}], "row_count": 1, "elapsed_ms": 42}',
            },
        }

        col1, col2 = st.columns([1, 1])
        with col1:
            chosen_tool = st.selectbox("Select a tool", list(TOOLS_DEMO.keys()))
            tool_info = TOOLS_DEMO[chosen_tool]
            st.markdown(f"**Description:** {tool_info['description']}")
            st.markdown("**Parameters:**")
            for pname, pdesc in tool_info['parameters'].items():
                st.markdown(f"- `{pname}`: {pdesc}")

        with col2:
            st.markdown("**Example tool call (JSON):**")
            st.code(tool_info['example_call'], language="json")
            st.markdown("**Tool result returned to agent:**")
            st.code(tool_info['example_result'], language="json")

        st.markdown("### Full message flow")
        flow_steps = [
            ("User message", "What's the weather in Sofia today?", "#f0f0f0", "#333"),
            ("LLM generates tool call", f'<tool_call>\n{{"name": "{chosen_tool}",\n "args": {tool_info["example_call"]}}}\n</tool_call>', "#fff8e1", "#b8860b"),
            ("Tool executes & returns", tool_info["example_result"], "#e8f5e9", "#1b5e20"),
            ("LLM generates final response", "Based on the tool result, the agent now composes a natural language answer to the user.", "#f3f0ff", "#534AB7"),
        ]
        for label, content_f, bg, col_f in flow_steps:
            st.markdown(f"""
            <div style="background:{bg};border-radius:8px;padding:0.7rem 1rem;
                        margin-bottom:8px;border-left:3px solid {col_f}">
                <div style="font-size:0.75rem;font-weight:600;color:{col_f};
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">
                    {label}</div>
                <code style="font-size:0.85rem;color:{col_f}">{content_f}</code>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### How the agent decides which tool to call")
        st.markdown("""
        Given a user message, the agent scores each available tool and picks the best fit.
        In practice this is done by the LLM reasoning about the tool descriptions.
        """)

        user_query = st.text_input("User query", "What is the square root of 1764?")

        # simple keyword scoring for demo
        tool_scores = {}
        q_lower = user_query.lower()
        tool_scores["get_weather"]      = sum(w in q_lower for w in ["weather","temperature","rain","sunny","forecast","climate","cold","hot"])
        tool_scores["web_search"]       = sum(w in q_lower for w in ["search","find","latest","news","who","what","when","where","research","recent"])
        tool_scores["code_interpreter"] = sum(w in q_lower for w in ["calculate","compute","code","run","python","math","sqrt","sum","plot","sort","algorithm"])
        tool_scores["database_query"]   = sum(w in q_lower for w in ["database","query","sql","table","rows","records","average","count","users","data"])

        total = sum(tool_scores.values()) or 1
        tool_probs = {k: v/total for k,v in tool_scores.items()}
        best_tool = max(tool_scores, key=tool_scores.get)

        fig = go.Figure(go.Bar(
            x=list(tool_scores.keys()),
            y=[tool_probs[k] for k in tool_scores],
            marker_color=['#534AB7' if k==best_tool else '#AFA9EC' for k in tool_scores],
            text=[f"{tool_probs[k]:.0%}" for k in tool_scores],
            textposition='outside'))
        fig.update_layout(yaxis_title="Relevance score", height=300, showlegend=False,
            yaxis_range=[0, 1.2])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Best tool for this query:** `{best_tool}`")
        st.caption("Real agents use the LLM itself to select tools based on natural language descriptions — "
                   "not keyword matching. This demo uses simple keyword heuristics for illustration.")


# ═══════════════════════════════════════════════════════════════════════════
# PLANNING & TASK DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "planning":
    st.title("🗺️ Planning & Task Decomposition")
    st.markdown("""
    <div class="concept-card">
    Complex tasks must be broken into <b>subtasks</b> before an agent can act.
    Good planning identifies <b>dependencies</b> between steps (what must happen before what),
    enables <b>parallelism</b> where possible, and handles failures gracefully.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Task dependency graph", "Planning strategies"])

    with tab1:
        TASK_PLANS = {
            "Write a research report": {
                "tasks": [
                    ("T1", "Define research question", []),
                    ("T2", "Search for sources",       ["T1"]),
                    ("T3", "Read & summarise source A",["T2"]),
                    ("T4", "Read & summarise source B",["T2"]),
                    ("T5", "Read & summarise source C",["T2"]),
                    ("T6", "Synthesise findings",      ["T3","T4","T5"]),
                    ("T7", "Write draft",              ["T6"]),
                    ("T8", "Review & edit",            ["T7"]),
                    ("T9", "Final report",             ["T8"]),
                ],
                "parallel": ["T3","T4","T5"],
            },
            "Build and deploy a feature": {
                "tasks": [
                    ("T1", "Write requirements", []),
                    ("T2", "Design schema",      ["T1"]),
                    ("T3", "Write backend code", ["T2"]),
                    ("T4", "Write frontend code",["T2"]),
                    ("T5", "Write tests",        ["T3","T4"]),
                    ("T6", "Code review",        ["T5"]),
                    ("T7", "Deploy to staging",  ["T6"]),
                    ("T8", "QA testing",         ["T7"]),
                    ("T9", "Deploy to prod",     ["T8"]),
                ],
                "parallel": ["T3","T4"],
            },
            "Plan a trip": {
                "tasks": [
                    ("T1", "Choose destination", []),
                    ("T2", "Check visa requirements",["T1"]),
                    ("T3", "Search flights",     ["T1"]),
                    ("T4", "Search hotels",      ["T1"]),
                    ("T5", "Book flights",       ["T3"]),
                    ("T6", "Book hotel",         ["T4"]),
                    ("T7", "Plan itinerary",     ["T5","T6"]),
                    ("T8", "Pack & prepare",     ["T2","T7"]),
                ],
                "parallel": ["T2","T3","T4"],
            },
        }

        col1, col2 = st.columns([1, 2])
        with col1:
            plan_choice = st.selectbox("Task scenario", list(TASK_PLANS.keys()))
            plan = TASK_PLANS[plan_choice]
            tasks = plan["tasks"]
            parallel = plan["parallel"]

            st.markdown("**Task list:**")
            for tid, tname, deps in tasks:
                dep_str = f" (after {', '.join(deps)})" if deps else " (start)"
                is_par  = "⚡ parallel" if tid in parallel else ""
                st.markdown(f"**{tid}** {tname}{dep_str} {is_par}")

        with col2:
            # layout tasks in levels (topological sort approximation)
            def compute_levels(tasks):
                levels = {}
                task_dict = {t[0]: t[2] for t in tasks}
                def level(tid):
                    if tid not in levels:
                        deps = task_dict[tid]
                        levels[tid] = (max(level(d) for d in deps) + 1) if deps else 0
                    return levels[tid]
                for t in tasks: level(t[0])
                return levels

            lvls = compute_levels(tasks)
            max_lvl = max(lvls.values())

            # group by level
            from collections import defaultdict
            by_level = defaultdict(list)
            for tid, tname, _ in tasks:
                by_level[lvls[tid]].append((tid, tname))

            fig = go.Figure()
            fig.update_layout(xaxis=dict(visible=False, range=[-0.5, max_lvl+0.5]),
                yaxis=dict(visible=False),
                height=420, plot_bgcolor="white",
                margin=dict(l=10,r=10,t=10,b=10))

            pos = {}
            for lvl in range(max_lvl+1):
                nodes_at = by_level[lvl]
                n = len(nodes_at)
                for i, (tid, tname) in enumerate(nodes_at):
                    y = (i - (n-1)/2) * 1.5
                    pos[tid] = (lvl, y)
                    is_par = tid in parallel
                    fill = "#fff3cd" if is_par else "#eeedfe"
                    border = "#EF9F27" if is_par else "#534AB7"
                    fig.add_shape(type="rect",
                        x0=lvl-0.38, y0=y-0.45, x1=lvl+0.38, y1=y+0.45,
                        fillcolor=fill, line=dict(color=border, width=2))
                    label = tname if len(tname)<=18 else tname[:16]+"…"
                    fig.add_annotation(x=lvl, y=y+0.12, text=f"<b>{tid}</b>",
                        showarrow=False, font=dict(size=10, color=border))
                    fig.add_annotation(x=lvl, y=y-0.18, text=label,
                        showarrow=False, font=dict(size=8, color="#444"))

            # draw dependency arrows
            for tid, tname, deps in tasks:
                x1, y1 = pos[tid]
                for dep in deps:
                    x0, y0 = pos[dep]
                    fig.add_shape(type="line", x0=x0+0.38, y0=y0, x1=x1-0.38, y1=y1,
                        line=dict(color="#aaa", width=1.2))

            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟡 Yellow = can run in parallel. Purple = sequential dependency.")

        # stats
        n_parallel = len(parallel)
        n_seq = len(tasks) - n_parallel
        critical = max_lvl + 1
        c1,c2,c3 = st.columns(3)
        c1.metric("Total subtasks", len(tasks))
        c2.metric("Parallelisable", n_parallel)
        c3.metric("Critical path length", critical)

    with tab2:
        st.markdown("### Planning strategies compared")
        import pandas as pd
        strategies = [
            ("Sequential",       "Execute subtasks one by one in order",
             "Simple, predictable", "Slow — no parallelism", "Simple linear tasks"),
            ("Hierarchical",     "Break goal → sub-goals → actions recursively",
             "Handles complexity", "Risk of over-planning", "Multi-step research, coding"),
            ("ReAct (reactive)", "Plan one step at a time based on observations",
             "Adapts to surprises", "May take more steps", "Web browsing, tool use"),
            ("Tree of Thought",  "Explore multiple reasoning branches, prune bad ones",
             "Better on hard problems", "High token cost", "Maths, strategy, puzzles"),
            ("Multi-agent",      "Assign subtasks to specialised sub-agents in parallel",
             "Fast, parallel", "Complex coordination", "Large pipelines, code generation"),
        ]
        st.dataframe(pd.DataFrame(strategies,
            columns=["Strategy","How it works","Advantage","Disadvantage","Best for"]),
            use_container_width=True, hide_index=True)

        st.markdown("### When does planning fail?")
        failure_modes = {
            "Incorrect decomposition": 0.30,
            "Missing dependencies": 0.20,
            "Tool call errors": 0.25,
            "Context window overflow": 0.15,
            "Infinite loops": 0.10,
        }
        fig2 = go.Figure(go.Bar(
            x=list(failure_modes.values()), y=list(failure_modes.keys()),
            orientation='h',
            marker_color=['#E24B4A','#EF9F27','#534AB7','#1D9E75','#D4537E'],
            text=[f"{v:.0%}" for v in failure_modes.values()],
            textposition='outside'))
        fig2.update_layout(xaxis_title="Relative frequency", height=280,
            xaxis_range=[0, 0.45], showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Illustrative distribution — real failure rates depend heavily on model capability and task complexity.")


# ═══════════════════════════════════════════════════════════════════════════
# AGENT MEMORY TYPES
# ═══════════════════════════════════════════════════════════════════════════
elif section == "agent_memory":
    st.title("🧠 Agent Memory Types")
    st.markdown("""
    <div class="concept-card">
    Agents need different kinds of memory to operate effectively.
    No single memory type covers all needs — real systems combine several.
    The right mix depends on the task horizon, latency requirements and cost.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Memory taxonomy", "RAG retrieval visualised"])

    with tab1:
        import pandas as pd
        memory_types = [
            ("In-context (working)", "Text in the current prompt window",
             "Instant, no setup", "Limited by context length, lost after session",
             "Current conversation, recent tool results", "🟢 Fast"),
            ("External / Vector store", "Embeddings in a vector DB (RAG)",
             "Scales to millions of docs, persistent",
             "Retrieval latency, embedding cost, may miss context",
             "Company knowledge bases, long-term user history", "🟡 Medium"),
            ("Parametric (weights)", "Knowledge baked into model weights during training",
             "Zero latency, always available",
             "Cannot be updated without retraining, may hallucinate",
             "World knowledge, language, reasoning", "🟢 Fast"),
            ("Episodic / Log store", "Structured record of past agent runs",
             "Full history, inspectable",
             "Grows indefinitely, needs summarisation",
             "Multi-session continuity, debugging", "🟡 Medium"),
            ("Cache (KV cache)", "Saved key-value pairs from previous forward passes",
             "Reduces repeated computation cost",
             "Memory-intensive, invalidated on change",
             "Long stable system prompts, repeated prefixes", "🟢 Fast"),
        ]
        st.dataframe(pd.DataFrame(memory_types,
            columns=["Type","What it is","Advantage","Limitation","Used for","Speed"]),
            use_container_width=True, hide_index=True)

        st.markdown("### Memory access pattern during a single agent turn")
        fig = go.Figure()
        fig.update_layout(xaxis=dict(visible=False, range=[0,10]),
            yaxis=dict(visible=False, range=[0,10]),
            height=320, plot_bgcolor="white",
            margin=dict(l=10,r=10,t=10,b=10))

        items = [
            (5, 9, "User Query", "#f3f0ff", "#534AB7"),
            (5, 7.2, "In-context window\n(system prompt + history)", "#e8f4ff", "#1565C0"),
            (2, 5, "Vector DB\n(RAG retrieval)", "#fff8e1", "#B8860B"),
            (8, 5, "Parametric\n(model weights)", "#e8f5e9", "#1B5E20"),
            (5, 3, "Agent reasoning\n& response", "#fdecea", "#B71C1C"),
        ]
        for x,y,label,fill,border in items:
            fig.add_shape(type="rect", x0=x-1.5, y0=y-0.7, x1=x+1.5, y1=y+0.7,
                fillcolor=fill, line=dict(color=border, width=2))
            for di,line in enumerate(label.split("\n")):
                fig.add_annotation(x=x, y=y+0.15-di*0.38, text=line,
                    showarrow=False, font=dict(size=10, color=border))

        for x0,y0,x1,y1 in [(5,8.3,5,7.9),(3.5,7.2,2,5.7),(6.5,7.2,8,5.7),
                              (2,4.3,4,3.7),(8,4.3,6,3.7),(5,2.3,5,1.5)]:
            fig.add_shape(type="line",x0=x0,y0=y0,x1=x1,y1=y1,
                line=dict(color="#aaa",width=1.5))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### RAG — Retrieval Augmented Generation")
        st.markdown("""
        RAG retrieves relevant documents from an external store at query time,
        injects them into the context, and lets the LLM answer with up-to-date grounded information.
        """)

        DOCS = [
            "Transformers use self-attention to model relationships between all tokens simultaneously.",
            "Gradient descent minimises the loss by moving parameters in the direction of steepest descent.",
            "Convolutional neural networks use sliding kernels to detect local patterns in images.",
            "RLHF aligns language models with human preferences using reward models and PPO.",
            "Vector databases store embeddings and support fast approximate nearest-neighbour search.",
            "LoRA fine-tunes large models efficiently by adding low-rank adapter matrices.",
            "The attention mechanism computes query-key dot products scaled by sqrt(d_k).",
            "Dropout randomly zeros activations during training to reduce overfitting.",
            "PCA projects data onto the directions of maximum variance using eigenvectors.",
            "The chain rule allows backpropagation to compute gradients through many layers.",
        ]

        col1, col2 = st.columns([1, 2])
        with col1:
            query_rag = st.text_input("Query", "How does attention work?")
            top_k = st.slider("Retrieve top-k docs", 1, 5, 3)
            st.markdown("**Document store (10 docs):**")
            for i, doc in enumerate(DOCS):
                st.markdown(f"<small>Doc {i+1}: {doc[:55]}…</small>", unsafe_allow_html=True)

        # simple TF-IDF-like scoring
        query_words = set(query_rag.lower().split())
        scores = []
        for doc in DOCS:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            # bonus for key phrase matches
            bonus = sum(2 for w in query_words if any(w in dw for dw in doc_words))
            scores.append(overlap + bonus * 0.3)

        scores_arr = np.array(scores, dtype=float)
        scores_norm = scores_arr / (scores_arr.max() + 1e-9)
        top_idx = np.argsort(scores_arr)[::-1]

        with col2:
            fig = go.Figure(go.Bar(
                x=[f"Doc {i+1}" for i in range(len(DOCS))],
                y=scores_norm,
                marker_color=['#534AB7' if i in top_idx[:top_k] else '#AFA9EC'
                              for i in range(len(DOCS))],
                text=[f"{s:.2f}" for s in scores_norm],
                textposition='outside'))
            fig.update_layout(yaxis_title="Similarity score", height=300,
                yaxis_range=[0,1.3], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Top {top_k} retrieved documents:**")
            for rank, idx in enumerate(top_idx[:top_k]):
                st.markdown(f"""
                <div style="background:#eeedfe;border-left:3px solid #534AB7;
                            padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;margin-bottom:6px">
                    <b>#{rank+1} Doc {idx+1}</b> (score={scores_norm[idx]:.2f})<br>
                    <small>{DOCS[idx]}</small>
                </div>
                """, unsafe_allow_html=True)
        st.caption("Real RAG uses dense vector embeddings (cosine similarity). "
                   "This demo uses simple keyword overlap for illustration.")


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-AGENT SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "multi_agent":
    st.title("🤝 Multi-Agent Systems")
    st.markdown("""
    <div class="concept-card">
    A <b>multi-agent system</b> uses several specialised AI agents working together.
    An <b>orchestrator</b> agent breaks down the task and routes subtasks to
    <b>worker agents</b>, each optimised for a specific capability.
    Results are aggregated and returned to the user.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Orchestrator pattern", "Agent communication"])

    with tab1:
        PIPELINES = {
            "Content creation pipeline": {
                "orchestrator": "Content Orchestrator",
                "workers": [
                    ("Research Agent",  "🔍", "Searches web, gathers facts and sources",           "#534AB7"),
                    ("Writer Agent",    "✍️", "Drafts content from research findings",             "#1D9E75"),
                    ("Editor Agent",    "📝", "Reviews, corrects grammar and improves clarity",    "#EF9F27"),
                    ("SEO Agent",       "📈", "Optimises keywords, meta tags, structure",          "#E24B4A"),
                    ("Publisher Agent", "🚀", "Formats and publishes to CMS",                     "#D4537E"),
                ],
                "flow": ["Research Agent","Writer Agent","Editor Agent","SEO Agent","Publisher Agent"],
                "parallel": ["Research Agent"],
            },
            "Software engineering pipeline": {
                "orchestrator": "Engineering Orchestrator",
                "workers": [
                    ("Planner Agent",   "🗺️", "Breaks feature into tasks and acceptance criteria", "#534AB7"),
                    ("Coder Agent",     "💻", "Writes implementation code",                       "#1D9E75"),
                    ("Test Agent",      "🧪", "Writes and runs unit/integration tests",            "#EF9F27"),
                    ("Review Agent",    "🔍", "Performs code review and security checks",         "#E24B4A"),
                    ("Deploy Agent",    "🚀", "Runs CI/CD pipeline and deploys to staging",       "#D4537E"),
                ],
                "flow": ["Planner Agent","Coder Agent","Test Agent","Review Agent","Deploy Agent"],
                "parallel": ["Coder Agent","Test Agent"],
            },
            "Data analysis pipeline": {
                "orchestrator": "Analysis Orchestrator",
                "workers": [
                    ("Ingestion Agent", "📥", "Fetches data from APIs and databases",             "#534AB7"),
                    ("Cleaning Agent",  "🧹", "Handles missing values, outliers, formatting",    "#1D9E75"),
                    ("Stats Agent",     "📊", "Computes descriptive statistics and correlations", "#EF9F27"),
                    ("ML Agent",        "🤖", "Trains and evaluates predictive models",           "#E24B4A"),
                    ("Report Agent",    "📄", "Generates charts, tables and narrative report",   "#D4537E"),
                ],
                "flow": ["Ingestion Agent","Cleaning Agent","Stats Agent","ML Agent","Report Agent"],
                "parallel": ["Stats Agent","ML Agent"],
            },
        }

        col1, col2 = st.columns([1, 2])
        with col1:
            pipeline_choice = st.selectbox("Pipeline", list(PIPELINES.keys()))
            pipeline = PIPELINES[pipeline_choice]
            active_step = st.slider("Simulate execution step",
                0, len(pipeline["workers"]), 2)
            st.markdown(f"**Orchestrator:** {pipeline['orchestrator']}")
            st.markdown("**Worker agents:**")
            for i, (name, icon, desc, col_w) in enumerate(pipeline["workers"]):
                status = "✅ done" if i < active_step else "⏳ waiting" if i == active_step else "⬜ queued"
                par = " ⚡parallel" if name in pipeline["parallel"] else ""
                st.markdown(f'<span style="color:{col_w}">●</span> **{icon} {name}** {status}{par}<br>'
                            f'<small style="color:#666">{desc}</small>', unsafe_allow_html=True)

        with col2:
            fig = go.Figure()
            fig.update_layout(xaxis=dict(visible=False, range=[-0.5, len(pipeline["workers"])+0.5]),
                yaxis=dict(visible=False, range=[-1, 4]),
                height=400, plot_bgcolor="white",
                margin=dict(l=10,r=10,t=10,b=10))

            # orchestrator at top
            fig.add_shape(type="rect", x0=len(pipeline["workers"])/2-1.2, y0=3.0,
                x1=len(pipeline["workers"])/2+1.2, y1=3.8,
                fillcolor="#f3f0ff", line=dict(color="#534AB7", width=2.5))
            fig.add_annotation(x=len(pipeline["workers"])/2, y=3.4,
                text=f"<b>🎯 {pipeline['orchestrator']}</b>",
                showarrow=False, font=dict(size=11, color="#534AB7"))

            workers = pipeline["workers"]
            xs = np.linspace(0.5, len(workers)-0.5, len(workers))
            for i, (name, icon, desc, col_w) in enumerate(workers):
                x = xs[i]
                is_done = i < active_step
                is_active = i == active_step
                fill = "#d4edda" if is_done else "#fff3cd" if is_active else "#f8f9fa"
                border = col_w
                lw = 3 if is_active else 2
                fig.add_shape(type="rect", x0=x-0.45, y0=0.8, x1=x+0.45, y1=2.2,
                    fillcolor=fill, line=dict(color=border, width=lw))
                fig.add_annotation(x=x, y=1.7, text=f"{icon}",
                    showarrow=False, font=dict(size=16))
                short_name = name.replace(" Agent","")
                fig.add_annotation(x=x, y=1.2, text=f"<b>{short_name}</b>",
                    showarrow=False, font=dict(size=9, color=col_w))
                status_icon = "✅" if is_done else "⚡" if is_active else "⬜"
                fig.add_annotation(x=x, y=0.95, text=status_icon,
                    showarrow=False, font=dict(size=10))

                # line from orchestrator to worker
                fig.add_shape(type="line", x0=len(workers)/2, y0=3.0,
                    x1=x, y1=2.2, line=dict(color="#aaa", width=1, dash="dot"))

                # arrows between sequential workers
                if i < len(workers)-1 and workers[i][0] not in pipeline["parallel"]:
                    fig.add_shape(type="line", x0=x+0.45, y0=1.5,
                        x1=xs[i+1]-0.45, y1=1.5,
                        line=dict(color=col_w, width=1.5))

            # result arrow back up
            fig.add_shape(type="line",
                x0=xs[-1], y0=2.2, x1=len(workers)/2, y1=3.0,
                line=dict(color="#1D9E75", width=2, dash="dash"))
            fig.add_annotation(x=len(workers)/2+0.8, y=2.6, text="result",
                showarrow=False, font=dict(size=9, color="#1D9E75"))

            st.plotly_chart(fig, use_container_width=True)
            st.caption("⚡ = parallel workers. Drag the slider to watch the pipeline execute.")

    with tab2:
        st.markdown("### How agents communicate")
        import pandas as pd
        comm_patterns = [
            ("Message passing", "Agents send structured messages (JSON/text) via a shared bus or queue",
             "Decoupled, async-friendly", "Needs schema design"),
            ("Shared memory / blackboard", "All agents read/write to a common state store",
             "Simple, visible to all", "Concurrency conflicts"),
            ("Function calls", "Orchestrator calls sub-agents as if they were tools",
             "Clean, hierarchical", "Sequential by default"),
            ("Streaming", "Agents pipe output tokens directly to the next agent",
             "Low latency", "Complex error handling"),
        ]
        st.dataframe(pd.DataFrame(comm_patterns,
            columns=["Pattern","Description","Pro","Con"]),
            use_container_width=True, hide_index=True)

        st.markdown("### Key design decisions")
        decisions = {
            "How many agents?": "Start with one. Add agents only when a clear specialisation boundary exists.",
            "Synchronous or async?": "Sync is simpler to debug. Async is needed for parallelism and long-running tasks.",
            "How to handle failure?": "Each agent should return success/error. Orchestrator retries or falls back.",
            "How to share context?": "Pass minimal context in each message. Use IDs to reference large objects in shared stores.",
            "How to avoid loops?": "Set a max iteration count. Track visited states. Use a supervisor agent.",
        }
        for q, a in decisions.items():
            with st.expander(q):
                st.markdown(a)

# ═══════════════════════════════════════════════════════════════════════════
# VECTOR NORMS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "vector_norms":
    st.title("‖·‖ Vector Norms")
    st.markdown("""
    <div class="concept-card">
    A <b>norm</b> is a function that assigns a non-negative <em>length</em> or <em>size</em> to a vector.
    Different norms measure size differently — and choosing the right one shapes how ML models learn,
    regularize, and measure distance.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Unit balls — geometry of norms", "Live norm calculator", "Connection to regularization"])

    # ── TAB 1: Unit balls ─────────────────────────────────────────────────
    with tab1:
        st.markdown("### What does 'all vectors of norm = 1' look like?")
        st.markdown(
            "The **unit ball** is the set of all vectors **v** with ‖v‖ = 1. "
            "Its shape reveals the norm's geometry."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            p_val = st.slider("p  (Lp norm)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
            show_l1 = st.checkbox("Overlay L1  (p=1)", value=True)
            show_l2 = st.checkbox("Overlay L2  (p=2)", value=True)
            st.markdown('<div class="formula-box">‖v‖ₚ = (|v₁|ᵖ + |v₂|ᵖ)^(1/p)</div>', unsafe_allow_html=True)
            if p_val < 1.0:
                st.warning("p < 1 is not a true norm (triangle inequality fails) but is used in sparse optimization.")
            elif p_val == 1.0:
                st.info("p = 1 → L1 norm. Diamond shape; promotes sparsity.")
            elif p_val == 2.0:
                st.success("p = 2 → L2 norm (Euclidean). Perfect circle.")
            elif p_val >= 6.0:
                st.info("p → ∞ → L∞ norm. Approaches a square: max(|v₁|, |v₂|).")
            else:
                st.info(f"p = {p_val:.1f} — intermediate shape between diamond and circle.")

        with col2:
            # Generate unit-ball contour for current p
            theta = np.linspace(0, 2 * np.pi, 800)
            # Parametric: sign(cos)*|cos|^(2/p), sign(sin)*|sin|^(2/p)
            def lp_unit_ball(p):
                t = np.linspace(0, 2 * np.pi, 800)
                x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / p)
                y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / p)
                return x, y

            fig = go.Figure()

            if show_l1:
                x1, y1 = lp_unit_ball(1.0)
                fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines",
                    line=dict(color="#E24B4A", width=1.5, dash="dash"),
                    name="L1 (p=1)"))
            if show_l2:
                x2, y2 = lp_unit_ball(2.0)
                fig.add_trace(go.Scatter(x=x2, y=y2, mode="lines",
                    line=dict(color="#1D9E75", width=1.5, dash="dot"),
                    name="L2 (p=2)"))

            xp, yp = lp_unit_ball(p_val)
            fig.add_trace(go.Scatter(x=xp, y=yp, mode="lines",
                line=dict(color="#534AB7", width=3),
                fill="toself", fillcolor="rgba(83,74,183,0.08)",
                name=f"Lp (p={p_val:.1f})"))

            fig.update_layout(
                height=400,
                xaxis=dict(range=[-1.5, 1.5], zeroline=True, zerolinecolor="#ccc",
                           scaleanchor="y", title="v₁"),
                yaxis=dict(range=[-1.5, 1.5], zeroline=True, zerolinecolor="#ccc",
                           title="v₂"),
                plot_bgcolor="white",
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key geometric intuitions:**
        - **L1 (diamond)** — corners touch the axes; this means a solution *on* the L1 unit ball is likely to sit at a corner, which has zero in all but one coordinate → **sparsity**.
        - **L2 (circle)** — perfectly symmetric; no preferred direction, so solutions shrink smoothly towards zero without going exactly to zero.
        - **L∞ (square)** — measures the *largest* single coordinate: ‖v‖∞ = max(|v₁|, |v₂|).
        - **p < 1** — the shape caves inward (non-convex); even sparser solutions but harder to optimise.
        """)

    # ── TAB 2: Live norm calculator ───────────────────────────────────────
    with tab2:
        st.markdown("### Move the vector — watch the norms update")

        col1, col2 = st.columns([1, 2])
        with col1:
            vx = st.slider("v₁", -5.0, 5.0, 3.0, step=0.1)
            vy = st.slider("v₂", -5.0, 5.0, 4.0, step=0.1)

            v = np.array([vx, vy])
            l0 = int(np.sum(v != 0))          # pseudo L0 (count of non-zeros)
            l1 = np.sum(np.abs(v))
            l2 = np.linalg.norm(v)
            linf = np.max(np.abs(v))

            st.markdown('<div class="formula-box">‖v‖₀  =  ' + str(l0) + '  (non-zero entries)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="formula-box">‖v‖₁  =  |{vx}| + |{vy}|  =  {l1:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="formula-box">‖v‖₂  =  √({vx}² + {vy}²)  =  {l2:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="formula-box">‖v‖∞  =  max(|{vx}|, |{vy}|)  =  {linf:.3f}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Relationship (always true):**")
            st.markdown(f"‖v‖∞ ≤ ‖v‖₂ ≤ ‖v‖₁  →  {linf:.2f} ≤ {l2:.2f} ≤ {l1:.2f}")

        with col2:
            fig = go.Figure()

            # Draw the norm "radius" circles / diamonds at the computed values
            def lp_ball_scaled(p, r):
                t = np.linspace(0, 2 * np.pi, 800)
                x = r * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / p)
                y = r * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / p)
                return x, y

            # L1 ball at radius l1
            x1b, y1b = lp_ball_scaled(1.0, l1)
            fig.add_trace(go.Scatter(x=x1b, y=y1b, mode="lines",
                line=dict(color="#E24B4A", width=1.5, dash="dash"),
                name=f"L1 ball r={l1:.2f}"))

            # L2 ball at radius l2
            x2b, y2b = lp_ball_scaled(2.0, l2)
            fig.add_trace(go.Scatter(x=x2b, y=y2b, mode="lines",
                line=dict(color="#1D9E75", width=1.5, dash="dot"),
                name=f"L2 ball r={l2:.2f}"))

            # The vector
            fig.add_trace(go.Scatter(x=[0, vx], y=[0, vy], mode="lines+markers",
                line=dict(color="#534AB7", width=3),
                marker=dict(size=[0, 10], color="#534AB7"),
                name=f"v = [{vx}, {vy}]"))

            # Projections to axes (L1 path)
            fig.add_trace(go.Scatter(
                x=[vx, vx, 0], y=[0, vy, vy],
                mode="lines", line=dict(color="#E24B4A", width=1.5, dash="dot"),
                showlegend=False))

            ax_range = max(abs(vx), abs(vy), l1) * 1.3 + 1
            fig.update_layout(
                height=420,
                xaxis=dict(range=[-ax_range, ax_range], zeroline=True,
                           zerolinecolor="#ccc", scaleanchor="y", title="v₁"),
                yaxis=dict(range=[-ax_range, ax_range], zeroline=True,
                           zerolinecolor="#ccc", title="v₂"),
                plot_bgcolor="white",
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The L1 ball (dashed red diamond) just touches the vector tip. "
                       "The L2 ball (dotted green circle) also passes through the tip. "
                       "Dotted red lines show the Manhattan path (|v₁| + |v₂|).")

    # ── TAB 3: Connection to regularization ──────────────────────────────
    with tab3:
        st.markdown("### Why the shape of the norm determines sparsity")
        st.markdown("""
        In regularized training, we minimize:
        """)
        st.markdown('<div class="formula-box">Loss(w) + λ · ‖w‖ₚ</div>', unsafe_allow_html=True)
        st.markdown("""
        The optimal solution is where the **loss contours** first touch the **norm ball**.
        The geometry of the ball determines whether that touching point has zeros.
        """)

        col1, col2 = st.columns([1, 2])
        with col1:
            lambda_reg = st.slider("Regularization strength λ", 0.01, 2.0, 0.5, step=0.05)
            norm_type  = st.radio("Norm", ["L1", "L2"], horizontal=True)
            loss_center_x = st.slider("Loss minimum  w₁*", -3.0, 3.0, 2.5, step=0.1)
            loss_center_y = st.slider("Loss minimum  w₂*", -3.0, 3.0, 2.0, step=0.1)
            p_reg = 1.0 if norm_type == "L1" else 2.0

        with col2:
            w1_grid = np.linspace(-3.5, 3.5, 300)
            w2_grid = np.linspace(-3.5, 3.5, 300)
            W1, W2  = np.meshgrid(w1_grid, w2_grid)

            # Quadratic loss centred at (loss_center_x, loss_center_y)
            Loss = (W1 - loss_center_x)**2 + 1.2*(W2 - loss_center_y)**2

            # Norm ball radius = 1/lambda (larger lambda → smaller allowed region)
            ball_r = 1.0 / lambda_reg

            fig = go.Figure()

            # Loss contours
            fig.add_trace(go.Contour(
                z=Loss, x=w1_grid, y=w2_grid,
                contours=dict(start=0.2, end=10, size=0.6, coloring="lines"),
                colorscale=[[0,"#c8e6c9"],[1,"#1D9E75"]],
                line=dict(width=1), showscale=False, name="Loss contours",
            ))

            # Norm ball boundary
            t = np.linspace(0, 2*np.pi, 800)
            bx = ball_r * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/p_reg)
            by = ball_r * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/p_reg)
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                line=dict(color="#E24B4A" if norm_type=="L1" else "#534AB7", width=2.5),
                name=f"{norm_type} ball  r={ball_r:.2f}"))

            # Find approximate touching point (constrained optimum)
            # Grid search inside ball
            mask = (np.abs(W1)**p_reg + np.abs(W2)**p_reg) <= ball_r**p_reg
            Loss_masked = np.where(mask, Loss, np.inf)
            idx = np.unravel_index(np.argmin(Loss_masked), Loss_masked.shape)
            opt_w1 = w1_grid[idx[1]]
            opt_w2 = w2_grid[idx[0]]

            fig.add_trace(go.Scatter(
                x=[opt_w1], y=[opt_w2], mode="markers",
                marker=dict(size=14, color="#EF9F27", symbol="star",
                            line=dict(color="#333", width=1.5)),
                name=f"Solution  ({opt_w1:.2f}, {opt_w2:.2f})"))

            # Loss minimum (unconstrained)
            fig.add_trace(go.Scatter(
                x=[loss_center_x], y=[loss_center_y], mode="markers",
                marker=dict(size=10, color="#888", symbol="x", line=dict(color="#555", width=2)),
                name="Unconstrained min"))

            fig.update_layout(
                height=420,
                xaxis=dict(range=[-3.5, 3.5], zeroline=True, zerolinecolor="#ccc",
                           scaleanchor="y", title="w₁"),
                yaxis=dict(range=[-3.5, 3.5], zeroline=True, zerolinecolor="#ccc", title="w₂"),
                plot_bgcolor="white",
                legend=dict(x=0.01, y=0.99, font=dict(size=11)),
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            if norm_type == "L1":
                if abs(opt_w1) < 0.05 or abs(opt_w2) < 0.05:
                    st.success(f"★ L1 solution sits at a **corner** of the diamond → one weight is ~0 (sparse!)  "
                               f"w = ({opt_w1:.2f}, {opt_w2:.2f})")
                else:
                    st.info(f"Solution: w = ({opt_w1:.2f}, {opt_w2:.2f}) — try moving the loss minimum further from the axes to see a corner solution.")
            else:
                st.info(f"★ L2 solution is on the **smooth circle** → both weights shrink but neither reaches exactly 0.  "
                        f"w = ({opt_w1:.2f}, {opt_w2:.2f})")

        st.markdown("""
        **Take-aways:**
        | | L1 norm | L2 norm |
        |---|---|---|
        | Penalty term | λ(|w₁| + |w₂|) | λ(w₁² + w₂²) |
        | Ball shape | Diamond (corners on axes) | Circle (smooth) |
        | Optimal solution | Likely at a corner → **sparse weights** | On the curve → **small but non-zero weights** |
        | Use when | You want feature selection | You want all features to contribute a little |
        | Also called | Lasso regularization | Ridge regularization |
        """)

# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "embeddings":
    st.title("🗺️ Embeddings")
    st.markdown("""
    <div class="concept-card">
    An <b>embedding</b> maps a discrete object — a word, sentence, image, user — into a
    dense vector of real numbers so that <em>similar things end up close together</em> in
    that space. The geometry of the space encodes meaning: direction, distance and angle
    all carry information the model can learn from.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Word2Vec vs GloVe", "Cosine similarity explorer", "Static vs contextual"])

    # ── TAB 1: Word2Vec vs GloVe ─────────────────────────────────────────
    with tab1:
        st.markdown("### How are embeddings trained?")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Word2Vec  *(Google, 2013)*")
            st.markdown("""
            **Core idea:** train a shallow neural network to *predict* words from their neighbours.
            The hidden-layer weights become the embedding vectors.

            Two flavours:
            - **Skip-gram** — given centre word, predict surrounding context words
            - **CBOW** — given context words, predict the centre word

            **Learning signal:** local context windows (e.g. ±2 words).
            The model never sees global corpus statistics — only one window at a time.
            """)
            st.markdown('<div class="formula-box">P(context | word) ≈ softmax(E_word · E_context^T)</div>', unsafe_allow_html=True)

            # Illustrate Skip-gram window
            sentence = ["the", "cat", "sat", "on", "the", "mat"]
            centre_idx = st.slider("Centre word (Skip-gram)", 1, 4, 2, key="w2v_centre")
            window = 2

            fig_sg = go.Figure()
            for i, word in enumerate(sentence):
                dist = abs(i - centre_idx)
                if i == centre_idx:
                    color, size, label = "#534AB7", 20, f"<b>{word}</b><br>(target)"
                elif dist <= window:
                    color, size, label = "#1D9E75", 14, f"{word}<br>(context)"
                else:
                    color, size, label = "#ccc", 10, word

                fig_sg.add_trace(go.Scatter(
                    x=[i], y=[0], mode="markers+text",
                    marker=dict(size=size, color=color),
                    text=[word], textposition="top center",
                    textfont=dict(color=color, size=12),
                    showlegend=False
                ))
                # window bracket lines
                if dist == window and dist > 0:
                    fig_sg.add_shape(type="line", x0=centre_idx, y0=-0.3, x1=i, y1=-0.3,
                        line=dict(color="#1D9E75", width=1.5, dash="dot"))

            fig_sg.update_layout(
                height=160, plot_bgcolor="white",
                xaxis=dict(visible=False, range=[-0.5, 5.5]),
                yaxis=dict(visible=False, range=[-0.6, 0.6]),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_sg, use_container_width=True)
            st.caption(f"Window = ±{window}. Purple = target word; green = context words used for training.")

        with col_b:
            st.markdown("#### GloVe  *(Stanford, 2014)*")
            st.markdown("""
            **Core idea:** factorize the *global* word co-occurrence matrix.
            Build a table of how often every word pair (i, j) appears within a window
            across the *entire* corpus, then find vectors whose dot product matches
            the log of that count.

            **Learning signal:** corpus-wide statistics — every sentence contributes
            to the same table before any vectors are learned.
            """)
            st.markdown('<div class="formula-box">w_i · w_j + b_i + b_j ≈ log X_ij</div>', unsafe_allow_html=True)
            st.markdown("*X_ij = co-occurrence count of words i and j*")

            # Co-occurrence matrix mini-demo
            words_demo = ["king", "queen", "man", "woman", "crown"]
            # Plausible toy co-occurrence counts
            co_matrix = np.array([
                [0,  15, 20,  8, 30],
                [15,  0,  8, 22, 28],
                [20,  8,  0, 35,  5],
                [ 8, 22, 35,  0,  4],
                [30, 28,  5,  4,  0],
            ], dtype=float)

            fig_co = go.Figure(data=go.Heatmap(
                z=co_matrix,
                x=words_demo, y=words_demo,
                colorscale="Blues",
                text=co_matrix.astype(int),
                texttemplate="%{text}",
                showscale=False,
            ))
            fig_co.update_layout(
                height=260, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Toy co-occurrence matrix X_ij", font=dict(size=12)),
            )
            st.plotly_chart(fig_co, use_container_width=True)
            st.caption("GloVe optimises vectors so w_i · w_j ≈ log(X_ij). High counts → similar vectors.")

        st.markdown("---")
        st.markdown("### Side-by-side comparison")
        import pandas as pd
        comparison = [
            ("Learning signal",    "Local context windows",                   "Global co-occurrence matrix"),
            ("Training objective", "Predict neighbour words (classification)", "Fit log co-occurrence counts (regression)"),
            ("Model type",         "Shallow neural network",                  "Weighted least-squares factorization"),
            ("Memory during train","Low — streams one window at a time",       "High — builds full N×N matrix first"),
            ("Strengths",          "Syntactic patterns; fast to train",        "Semantic analogies; leverages full corpus"),
            ("Weaknesses",         "Misses long-range corpus statistics",      "Memory-intensive for large vocabularies"),
            ("Both share",         "Dense fixed-size vectors per word",        "Dense fixed-size vectors per word"),
            ("Both share",         "Semantic arithmetic (king−man+woman≈queen)", "Semantic arithmetic (king−man+woman≈queen)"),
            ("Both share",         "Static — one vector per word regardless of context", "Static — one vector per word regardless of context"),
        ]
        df_cmp = pd.DataFrame(comparison, columns=["Aspect", "Word2Vec", "GloVe"])
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # ── TAB 2: Cosine similarity explorer ────────────────────────────────
    with tab2:
        st.markdown("### Cosine similarity — meaning as angle")
        st.markdown("""
        The standard way to compare two embeddings is **cosine similarity** — the cosine
        of the angle between them. It ignores magnitude (a long and a short vector pointing
        the same way are equally similar).
        """)
        st.markdown('<div class="formula-box">cos(θ) = (a · b) / (‖a‖ · ‖b‖)  ∈  [−1, 1]</div>', unsafe_allow_html=True)

        # Toy 2-D word embeddings (hand-crafted to illustrate clusters)
        toy_words = {
            "king":   np.array([ 2.1,  1.8]),
            "queen":  np.array([ 1.9,  2.2]),
            "man":    np.array([ 1.8, -0.3]),
            "woman":  np.array([ 1.6,  0.2]),
            "prince": np.array([ 2.3,  1.2]),
            "dog":    np.array([-1.5, -1.8]),
            "cat":    np.array([-1.8, -1.4]),
            "puppy":  np.array([-1.3, -2.1]),
            "run":    np.array([-0.5,  2.2]),
            "walk":   np.array([-0.3,  1.9]),
            "sprint": np.array([-0.7,  2.5]),
        }

        word_list = list(toy_words.keys())

        col1, col2 = st.columns([1, 2])
        with col1:
            word_a = st.selectbox("Word A", word_list, index=0)
            word_b = st.selectbox("Word B", word_list, index=1)

            va = toy_words[word_a]
            vb = toy_words[word_b]
            cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
            angle   = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

            st.metric("Cosine similarity", f"{cos_sim:.3f}")
            st.metric("Angle between vectors", f"{angle:.1f}°")

            if cos_sim > 0.95:
                st.success("Nearly identical direction — very similar meaning.")
            elif cos_sim > 0.7:
                st.success("High similarity — related concepts.")
            elif cos_sim > 0.3:
                st.info("Moderate similarity — loosely related.")
            elif cos_sim > 0:
                st.warning("Weak similarity — different topics.")
            else:
                st.error("Negative similarity — opposite directions.")

            st.markdown("---")
            st.markdown("**All similarities to Word A:**")
            sims = {w: np.dot(va, toy_words[w]) / (np.linalg.norm(va) * np.linalg.norm(toy_words[w]))
                    for w in word_list if w != word_a}
            for w, s in sorted(sims.items(), key=lambda x: -x[1]):
                bar = "█" * int((s + 1) * 10)
                st.markdown(f"`{w:<8}` {s:+.2f}  {bar}")

        with col2:
            fig = go.Figure()

            # Plot all word vectors as points
            clusters = {
                "royalty": ["king","queen","prince"],
                "gender":  ["man","woman"],
                "animals": ["dog","cat","puppy"],
                "motion":  ["run","walk","sprint"],
            }
            cluster_colors = {"royalty":"#534AB7","gender":"#D4537E","animals":"#1D9E75","motion":"#EF9F27"}

            for cluster, members in clusters.items():
                xs = [toy_words[w][0] for w in members]
                ys = [toy_words[w][1] for w in members]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers+text",
                    text=members, textposition="top center",
                    marker=dict(size=12, color=cluster_colors[cluster]),
                    name=cluster, textfont=dict(size=11)
                ))

            # Highlight selected pair with arrows from origin
            for word, color, label in [(word_a, "#222", "A"), (word_b, "#E24B4A", "B")]:
                v = toy_words[word]
                fig.add_annotation(
                    x=v[0], y=v[1], ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3, arrowwidth=2.5, arrowcolor=color,
                    text=f"<b>{label}</b>", font=dict(color=color, size=13),
                )

            # Arc for angle
            theta_a = np.arctan2(va[1], va[0])
            theta_b = np.arctan2(vb[1], vb[0])
            arc_r   = 0.5
            arc_t   = np.linspace(min(theta_a, theta_b), max(theta_a, theta_b), 50)
            fig.add_trace(go.Scatter(
                x=arc_r * np.cos(arc_t),
                y=arc_r * np.sin(arc_t),
                mode="lines", line=dict(color="#999", width=1.5, dash="dot"),
                showlegend=False
            ))

            fig.update_layout(
                height=420,
                xaxis=dict(range=[-2.8, 3.2], zeroline=True, zerolinecolor="#ddd", title="dim 1"),
                yaxis=dict(range=[-2.8, 3.2], zeroline=True, zerolinecolor="#ddd",
                           title="dim 2", scaleanchor="x"),
                plot_bgcolor="white",
                legend=dict(x=0.01, y=0.01),
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Toy 2-D embeddings — clusters show semantic grouping. "
                       "Arrows show the two selected words; the dotted arc is the angle between them.")

        st.markdown("### The famous analogy: king − man + woman ≈ queen")
        v_analogy = toy_words["king"] - toy_words["man"] + toy_words["woman"]
        sims_analogy = {w: np.dot(v_analogy, toy_words[w]) / (np.linalg.norm(v_analogy) * np.linalg.norm(toy_words[w]))
                        for w in word_list}
        top3 = sorted(sims_analogy.items(), key=lambda x: -x[1])[:3]
        st.markdown(f"In this toy space: **king − man + woman** is closest to → "
                    f"**{top3[0][0]}** (sim={top3[0][1]:.2f}), "
                    f"{top3[1][0]} ({top3[1][1]:.2f}), "
                    f"{top3[2][0]} ({top3[2][1]:.2f})")

    # ── TAB 3: Static vs Contextual ──────────────────────────────────────
    with tab3:
        st.markdown("### The fundamental limitation of Word2Vec and GloVe")
        st.markdown("""
        Both Word2Vec and GloVe produce **one fixed vector per word** — regardless of context.
        The word *"bank"* gets the same vector whether you mean a river bank or a financial bank.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Static embeddings (Word2Vec / GloVe)")
            st.markdown("""
            - One vector per word, learned once, frozen
            - Fast and memory-efficient at inference
            - Cannot distinguish word sense from context
            - Good for: bag-of-words models, similarity search, recommendation
            """)
            st.markdown('<div class="formula-box">embed("bank") → [0.3, −0.7, 1.2, ...]<br><small>same vector in every sentence</small></div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### Contextual embeddings (BERT, GPT, etc.)")
            st.markdown("""
            - A different vector per token *per context*
            - Computed by a full transformer forward pass
            - Captures polysemy, syntax, long-range dependencies
            - Good for: NLU, QA, classification, generation
            """)
            st.markdown('<div class="formula-box">embed("bank", "river bank") → [0.1, 0.9, −0.2, ...]<br>embed("bank", "savings bank") → [0.8, −0.1, 0.5, ...]</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Evolution of embeddings")

        timeline = [
            ("2003", "Neural LM\n(Bengio et al.)", "First dense word vectors as by-product of language modelling", "#ccc"),
            ("2013", "Word2Vec\n(Mikolov, Google)", "Efficient skip-gram & CBOW; popularised word embeddings", "#534AB7"),
            ("2014", "GloVe\n(Pennington, Stanford)", "Global co-occurrence matrix factorization", "#534AB7"),
            ("2017", "ELMo\n(Peters et al.)", "First contextual embeddings from bi-directional LSTMs", "#1D9E75"),
            ("2018", "BERT\n(Devlin, Google)", "Transformer encoder; masked language modelling; fine-tunable", "#1D9E75"),
            ("2019+", "Sentence-BERT,\nOpenAI Ada, etc.", "Optimised for sentence-level similarity and retrieval", "#EF9F27"),
        ]

        fig_tl = go.Figure()
        fig_tl.update_layout(
            height=280,
            xaxis=dict(visible=False, range=[-0.5, len(timeline) - 0.5]),
            yaxis=dict(visible=False, range=[-1, 3]),
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
        )

        for i, (year, name, desc, color) in enumerate(timeline):
            fig_tl.add_shape(type="circle",
                x0=i-0.35, y0=0.6, x1=i+0.35, y1=1.4,
                fillcolor=color, line=dict(color=color))
            fig_tl.add_annotation(x=i, y=1.0, text=f"<b>{year}</b>",
                showarrow=False, font=dict(size=10, color="white"))
            fig_tl.add_annotation(x=i, y=0.2, text=name.replace("\n","<br>"),
                showarrow=False, font=dict(size=9, color="#333"), align="center")
            fig_tl.add_annotation(x=i, y=2.2, text=desc,
                showarrow=False, font=dict(size=8, color="#555"), align="center",
                width=130)
            if i < len(timeline) - 1:
                fig_tl.add_shape(type="line", x0=i+0.35, y0=1.0,
                    x1=i+0.65, y1=1.0, line=dict(color="#aaa", width=2))

        st.plotly_chart(fig_tl, use_container_width=True)

        st.markdown("""
        **When to use which:**
        | Use case | Best choice |
        |---|---|
        | Large-scale similarity search / ANN index | Static (Word2Vec, GloVe, Ada) |
        | Text classification with limited compute | Static + simple classifier |
        | Named entity recognition, QA, NLU | Contextual (BERT family) |
        | Sentence / document retrieval (RAG) | Sentence-BERT, Ada-002, or similar |
        | Generation tasks | GPT-style contextual (decoder) |
        """)

# ═══════════════════════════════════════════════════════════════════════════
# VECTOR SPACES
# ═══════════════════════════════════════════════════════════════════════════
elif section == "vector_spaces":
    st.title("🧭 Vector Spaces")
    st.markdown("""
    <div class="concept-card">
    A <b>vector space</b> is the mathematical stage on which ML happens.
    Understanding <em>basis</em>, <em>span</em>, <em>projection</em> and <em>orthogonality</em>
    gives you the intuition behind PCA, attention heads, embeddings and least-squares — 
    all of which are fundamentally about finding the right directions in a space.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Unit vectors & normalisation",
        "Basis & span",
        "Projection",
        "Orthogonality & orthonormality",
    ])

    # ── TAB 1: Unit vectors ───────────────────────────────────────────────
    with tab1:
        st.markdown("### Unit vectors — stripping away magnitude, keeping direction")
        st.markdown("""
        A **unit vector** has magnitude exactly 1. It encodes pure direction.
        Normalising a vector means dividing it by its own magnitude to produce its unit vector.
        """)
        st.markdown('<div class="formula-box">v̂ = v / ‖v‖₂     so that  ‖v̂‖₂ = 1</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            vx = st.slider("v₁", -4.0, 4.0, 3.0, step=0.1, key="uv_x")
            vy = st.slider("v₂", -4.0, 4.0, 2.0, step=0.1, key="uv_y")
            mag = np.sqrt(vx**2 + vy**2)
            ux, uy = (vx / mag, vy / mag) if mag > 1e-9 else (0.0, 0.0)

            st.markdown("**Original vector v:**")
            st.markdown(f'<div class="formula-box">[{vx:.2f}, {vy:.2f}]<br>‖v‖ = {mag:.3f}</div>', unsafe_allow_html=True)
            st.markdown("**Unit vector v̂:**")
            st.markdown(f'<div class="formula-box">[{ux:.3f}, {uy:.3f}]<br>‖v̂‖ = {np.sqrt(ux**2+uy**2):.3f}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Standard basis unit vectors:**")
            st.markdown('<div class="formula-box">ê₁ = [1, 0]&nbsp;&nbsp;&nbsp;ê₂ = [0, 1]</div>', unsafe_allow_html=True)
            st.caption("Every vector is a linear combination of the standard basis: v = v₁ê₁ + v₂ê₂")

        with col2:
            fig = go.Figure()
            lim = max(abs(vx), abs(vy), 1.5) + 0.8

            # Unit circle
            theta_c = np.linspace(0, 2*np.pi, 200)
            fig.add_trace(go.Scatter(
                x=np.cos(theta_c), y=np.sin(theta_c),
                mode="lines", line=dict(color="#ddd", width=1.5, dash="dot"),
                name="Unit circle  ‖v‖=1", showlegend=True
            ))

            # Original vector
            fig.add_annotation(x=vx, y=vy, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor="#534AB7",
                text=f"<b>v</b> ‖v‖={mag:.2f}", font=dict(color="#534AB7", size=12),
                bgcolor="rgba(255,255,255,0.7)"
            )

            # Unit vector
            fig.add_annotation(x=ux, y=uy, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowwidth=2.5, arrowcolor="#1D9E75",
                text=f"<b>v̂</b> ‖v̂‖=1", font=dict(color="#1D9E75", size=12),
                bgcolor="rgba(255,255,255,0.7)"
            )

            # Standard basis
            for ex, ey, label in [(1,0,"ê₁"),(0,1,"ê₂")]:
                fig.add_annotation(x=ex, y=ey, ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#EF9F27",
                    text=label, font=dict(color="#EF9F27", size=12),
                )

            fig.update_layout(
                height=420,
                xaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc",
                           scaleanchor="y", title="v₁"),
                yaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc", title="v₂"),
                plot_bgcolor="white", legend=dict(x=0.01, y=0.99),
                margin=dict(l=40,r=20,t=20,b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Purple = original vector v. Green = unit vector v̂ (always touches unit circle). Orange = standard basis vectors.")

        st.markdown("""
        **Why unit vectors matter in ML:**
        - **Cosine similarity** compares embeddings as unit vectors — magnitude is normalised away so only direction counts
        - **Attention** in transformers projects queries and keys, then computes dot products — the scale-dot-product
          uses √d to keep unit-scale variance
        - **Weight initialisation** schemes (Xavier, He) ensure weight vectors start near unit scale
        """)

    # ── TAB 2: Basis & span ───────────────────────────────────────────────
    with tab2:
        st.markdown("### Basis vectors and the span they define")
        st.markdown("""
        A **basis** is a minimal set of vectors that can reach every point in the space
        through linear combinations. The **span** of a set of vectors is all the points
        you can reach by scaling and adding them.
        """)
        st.markdown('<div class="formula-box">span{b₁, b₂} = { α·b₁ + β·b₂  |  α, β ∈ ℝ }</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Basis vector b₁:**")
            b1x = st.slider("b₁ₓ", -3.0, 3.0, 1.0, step=0.1, key="bs_b1x")
            b1y = st.slider("b₁ᵧ", -3.0, 3.0, 0.0, step=0.1, key="bs_b1y")
            st.markdown("**Basis vector b₂:**")
            b2x = st.slider("b₂ₓ", -3.0, 3.0, 0.0, step=0.1, key="bs_b2x")
            b2y = st.slider("b₂ᵧ", -3.0, 3.0, 1.0, step=0.1, key="bs_b2y")
            st.markdown("**Target vector v = α·b₁ + β·b₂:**")
            alpha = st.slider("α (weight on b₁)", -3.0, 3.0, 1.5, step=0.1, key="bs_alpha")
            beta  = st.slider("β (weight on b₂)", -3.0, 3.0, 1.0, step=0.1, key="bs_beta")

            # Check linear independence
            det = b1x*b2y - b1y*b2x
            vx_t = alpha*b1x + beta*b2x
            vy_t = alpha*b1y + beta*b2y

            st.markdown(f'<div class="formula-box">v = {alpha}·b₁ + {beta}·b₂<br>= [{vx_t:.2f}, {vy_t:.2f}]</div>', unsafe_allow_html=True)

            if abs(det) < 0.05:
                st.error("⚠️ det ≈ 0 — these vectors are linearly dependent! They only span a line, not the full 2D plane.")
            else:
                st.success(f"✓ Linearly independent (det = {det:.2f}) — they span all of ℝ².")

        with col2:
            fig = go.Figure()
            lim = 4.5

            # Shade the span region (parallelogram grid)
            grid_a = np.linspace(-3, 3, 10)
            grid_b = np.linspace(-3, 3, 10)
            for a in grid_a:
                xs = [a*b1x + g*b2x for g in [-3,3]]
                ys = [a*b1y + g*b2y for g in [-3,3]]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                    line=dict(color="rgba(83,74,183,0.12)", width=1), showlegend=False))
            for b in grid_b:
                xs = [g*b1x + b*b2x for g in [-3,3]]
                ys = [g*b1y + b*b2y for g in [-3,3]]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                    line=dict(color="rgba(83,74,183,0.12)", width=1), showlegend=False))

            # Basis vectors
            for (bvx, bvy, label, color) in [(b1x,b1y,"b₁","#534AB7"),(b2x,b2y,"b₂","#E24B4A")]:
                fig.add_annotation(x=bvx, y=bvy, ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=color,
                    text=f"<b>{label}</b>", font=dict(color=color, size=13))

            # Parallelogram components
            fig.add_trace(go.Scatter(
                x=[0, alpha*b1x, vx_t],
                y=[0, alpha*b1y, vy_t],
                mode="lines", line=dict(color="#534AB7", width=1.5, dash="dot"), showlegend=False))
            fig.add_trace(go.Scatter(
                x=[0, beta*b2x, vx_t],
                y=[0, beta*b2y, vy_t],
                mode="lines", line=dict(color="#E24B4A", width=1.5, dash="dot"), showlegend=False))

            # Result vector
            fig.add_annotation(x=vx_t, y=vy_t, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowwidth=2.5, arrowcolor="#1D9E75",
                text=f"<b>v</b> [{vx_t:.1f},{vy_t:.1f}]", font=dict(color="#1D9E75", size=12))

            fig.update_layout(
                height=430,
                xaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc", scaleanchor="y", title="x"),
                yaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc", title="y"),
                plot_bgcolor="white", showlegend=False,
                margin=dict(l=40,r=20,t=20,b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Grid lines show the coordinate system defined by b₁ and b₂. "
                       "The green vector v is reached by stepping α times along b₁ and β times along b₂.")

        st.markdown("""
        **In ML:**
        - **PCA** finds a new basis aligned with the directions of maximum variance
        - **Embeddings** live in a high-dimensional vector space — the basis is learned, not hand-crafted
        - **Linear dependence** means redundant features — a sign of multicollinearity in regression
        """)

    # ── TAB 3: Projection ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### Projecting one vector onto another")
        st.markdown("""
        The **projection** of **a** onto **b** is the shadow **a** casts along the direction of **b**.
        It tells you: *how much of **a** lies in the direction of **b**?*
        """)
        st.markdown('<div class="formula-box">proj_b(a) = (a·b / ‖b‖²) · b = (a·b̂) · b̂</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            ax_p = st.slider("a₁", -4.0, 4.0,  3.0, step=0.1, key="pr_ax")
            ay_p = st.slider("a₂", -4.0, 4.0,  2.5, step=0.1, key="pr_ay")
            bx_p = st.slider("b₁", -4.0, 4.0,  4.0, step=0.1, key="pr_bx")
            by_p = st.slider("b₂", -4.0, 4.0,  1.0, step=0.1, key="pr_by")

            a = np.array([ax_p, ay_p])
            b = np.array([bx_p, by_p])
            b_norm2 = np.dot(b, b)

            if b_norm2 < 1e-9:
                st.warning("b is the zero vector — projection undefined.")
                scalar_proj = 0.0
                proj_vec = np.array([0.0, 0.0])
                perp_vec = a.copy()
            else:
                scalar_proj = np.dot(a, b) / np.sqrt(b_norm2)
                proj_vec = (np.dot(a, b) / b_norm2) * b
                perp_vec = a - proj_vec

            st.markdown("**Scalar projection** (signed length along b̂):")
            st.markdown(f'<div class="formula-box">a·b̂ = {scalar_proj:.3f}</div>', unsafe_allow_html=True)
            st.markdown("**Vector projection** (the actual shadow vector):")
            st.markdown(f'<div class="formula-box">proj_b(a) = [{proj_vec[0]:.3f}, {proj_vec[1]:.3f}]</div>', unsafe_allow_html=True)
            st.markdown("**Perpendicular component** a⊥:")
            st.markdown(f'<div class="formula-box">a⊥ = [{perp_vec[0]:.3f}, {perp_vec[1]:.3f}]</div>', unsafe_allow_html=True)
            st.caption("a = proj_b(a) + a⊥   (decomposition is always exact)")

        with col2:
            fig = go.Figure()
            lim = max(abs(ax_p), abs(ay_p), abs(bx_p), abs(by_p)) + 1.2

            # b vector (extended as reference line)
            b_hat = b / (np.sqrt(b_norm2) + 1e-9)
            fig.add_trace(go.Scatter(
                x=[-lim*b_hat[0]*0.9, lim*b_hat[0]*0.9],
                y=[-lim*b_hat[1]*0.9, lim*b_hat[1]*0.9],
                mode="lines", line=dict(color="#E24B4A", width=1, dash="dot"),
                showlegend=False
            ))

            # a vector
            fig.add_annotation(x=ax_p, y=ay_p, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor="#534AB7",
                text="<b>a</b>", font=dict(color="#534AB7", size=14))

            # b vector
            fig.add_annotation(x=bx_p, y=by_p, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor="#E24B4A",
                text="<b>b</b>", font=dict(color="#E24B4A", size=14))

            # projection vector
            fig.add_annotation(x=proj_vec[0], y=proj_vec[1], ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowwidth=2.5, arrowcolor="#1D9E75",
                text="<b>proj</b>", font=dict(color="#1D9E75", size=12))

            # perpendicular drop line (from a to proj)
            fig.add_trace(go.Scatter(
                x=[proj_vec[0], ax_p],
                y=[proj_vec[1], ay_p],
                mode="lines", line=dict(color="#EF9F27", width=2, dash="dash"),
                name="a⊥ (perpendicular)"
            ))

            # right-angle marker at proj_vec
            if b_norm2 > 1e-9:
                perp_dir = np.array([-b_hat[1], b_hat[0]]) * 0.25
                corner = proj_vec + perp_dir
                fig.add_trace(go.Scatter(
                    x=[proj_vec[0], corner[0], corner[0]+b_hat[0]*0.25],
                    y=[proj_vec[1], corner[1], corner[1]+b_hat[1]*0.25],
                    mode="lines", line=dict(color="#aaa", width=1.5),
                    showlegend=False
                ))

            fig.update_layout(
                height=430,
                xaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc",
                           scaleanchor="y", title="x"),
                yaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc", title="y"),
                plot_bgcolor="white",
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=40,r=20,t=20,b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green = projection of a onto b (the shadow). "
                       "Orange dashed = perpendicular component a⊥. Right-angle marker confirms they are perpendicular.")

        st.markdown("""
        **Why projection is everywhere in ML:**
        | Concept | What's being projected |
        |---|---|
        | **PCA** | Data points projected onto principal components (eigenvectors) |
        | **Least-squares regression** | Target y projected onto the column space of X |
        | **Attention** | Queries projected onto key directions to get attention scores |
        | **Cosine similarity** | Measures how much one embedding projects onto another |
        | **Gram-Schmidt** | Builds orthogonal basis by repeatedly subtracting projections |
        """)

    # ── TAB 4: Orthogonality & orthonormality ─────────────────────────────
    with tab4:
        st.markdown("### Orthogonality — vectors at right angles")
        st.markdown("""
        Two vectors are **orthogonal** if their dot product is zero — they share no component
        in common. A set of vectors is **orthonormal** if they are *all orthogonal to each other*
        and *each has magnitude 1*.
        """)

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown('<div class="formula-box">a ⊥ b  ⟺  a · b = 0</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Orthonormal:  bᵢ · bⱼ = δᵢⱼ<br><small>(1 if i=j, 0 if i≠j)</small></div>', unsafe_allow_html=True)
        with col2:
            st.markdown("""
            **Orthogonal** → dot product = 0, magnitudes can be anything

            **Orthonormal** → dot product = 0 AND each vector is a unit vector

            The standard basis {ê₁=[1,0], ê₂=[0,1]} is orthonormal.
            """)

        st.markdown("---")
        st.markdown("### Interactive: build an orthogonal pair")
        col1, col2 = st.columns([1,2])
        with col1:
            ox = st.slider("v₁", -3.0, 3.0, 2.0, step=0.1, key="orth_x")
            oy = st.slider("v₂", -3.0, 3.0, 1.0, step=0.1, key="orth_y")

            v1 = np.array([ox, oy])
            # Perpendicular vector is always [-oy, ox]
            v2 = np.array([-oy, ox])
            v1_unit = v1 / (np.linalg.norm(v1) + 1e-9)
            v2_unit = v2 / (np.linalg.norm(v2) + 1e-9)
            dot_check = np.dot(v1, v2)

            st.markdown("**v** (your vector):")
            st.markdown(f'<div class="formula-box">[{ox:.2f}, {oy:.2f}]  ‖v‖={np.linalg.norm(v1):.3f}</div>', unsafe_allow_html=True)
            st.markdown("**v⊥** (auto-computed orthogonal):")
            st.markdown(f'<div class="formula-box">[{v2[0]:.2f}, {v2[1]:.2f}]  ‖v⊥‖={np.linalg.norm(v2):.3f}</div>', unsafe_allow_html=True)
            st.markdown("**Dot product check:**")
            st.markdown(f'<div class="formula-box">v · v⊥ = {dot_check:.6f} ≈ 0 ✓</div>', unsafe_allow_html=True)
            st.markdown("**Unit vectors (orthonormal pair):**")
            st.markdown(f'<div class="formula-box">û = [{v1_unit[0]:.3f}, {v1_unit[1]:.3f}]<br>û⊥ = [{v2_unit[0]:.3f}, {v2_unit[1]:.3f}]</div>', unsafe_allow_html=True)

        with col2:
            fig = go.Figure()
            lim = max(abs(ox), abs(oy), 1.5) + 1.0

            # Unit circle
            theta_c = np.linspace(0, 2*np.pi, 200)
            fig.add_trace(go.Scatter(x=np.cos(theta_c), y=np.sin(theta_c),
                mode="lines", line=dict(color="#eee", width=1.5), showlegend=False))

            # v and v⊥ (original)
            for vec, label, color in [(v1,"v","#534AB7"),(v2,"v⊥","#E24B4A")]:
                fig.add_annotation(x=vec[0], y=vec[1], ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=color,
                    text=f"<b>{label}</b>", font=dict(color=color, size=14))

            # Unit versions
            for vec, label, color in [(v1_unit,"û","#534AB7"),(v2_unit,"û⊥","#E24B4A")]:
                fig.add_annotation(x=vec[0], y=vec[1], ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowwidth=1.5,
                    arrowcolor=color,
                    line=dict(dash="dot"),
                    text=f"<b>{label}</b>", font=dict(color=color, size=11))

            # right-angle marker at origin
            scale = 0.25
            corner = v1_unit * scale + v2_unit * scale
            fig.add_trace(go.Scatter(
                x=[v1_unit[0]*scale, corner[0], v2_unit[0]*scale],
                y=[v1_unit[1]*scale, corner[1], v2_unit[1]*scale],
                mode="lines", line=dict(color="#aaa", width=1.5), showlegend=False
            ))

            fig.update_layout(
                height=430,
                xaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc",
                           scaleanchor="y", title="x"),
                yaxis=dict(range=[-lim,lim], zeroline=True, zerolinecolor="#ccc", title="y"),
                plot_bgcolor="white", showlegend=False,
                margin=dict(l=40,r=20,t=20,b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Solid arrows = original vectors. Dashed = unit versions (touching unit circle). "
                       "Right-angle marker at origin confirms orthogonality.")

        st.markdown("---")
        st.markdown("### Why orthogonality is the backbone of ML")

        cols = st.columns(3)
        with cols[0]:
            st.markdown("#### PCA")
            st.markdown("""
            Principal components are **orthogonal** by construction — each new axis is perpendicular
            to all previous ones, so there is zero redundancy between components.
            """)
        with cols[1]:
            st.markdown("#### Attention heads")
            st.markdown("""
            Multiple attention heads learn **different** subspaces. Near-orthogonal heads
            capture different aspects of meaning — syntax in one, semantics in another.
            """)
        with cols[2]:
            st.markdown("#### Weight matrices (QR)")
            st.markdown("""
            Orthogonal weight initialisation preserves gradient norms during backprop —
            a key technique for training deep networks without vanishing gradients.
            """)

        st.markdown("""
        **Gram-Schmidt process** — how to turn any set of linearly independent vectors into an orthonormal basis:
        """)
        st.markdown('<div class="formula-box">u₁ = v₁ / ‖v₁‖<br>u₂ = (v₂ − (v₂·u₁)u₁) / ‖…‖<br>u₃ = (v₃ − (v₃·u₁)u₁ − (v₃·u₂)u₂) / ‖…‖  …</div>', unsafe_allow_html=True)
        st.caption("At each step: subtract the projections onto all previous basis vectors, then normalise. "
                   "This is exactly what QR decomposition does algorithmically.")

# ═══════════════════════════════════════════════════════════════════════════
# RECURRENT NETWORKS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "rnn":
    st.title("🔁 Recurrent Networks")
    st.markdown("""
    <div class="concept-card">
    A <b>recurrent network</b> processes sequences by maintaining a <em>hidden state</em>
    — a memory that carries information forward through time. Unlike feedforward networks
    which treat each input independently, RNNs share the same weights at every timestep,
    allowing them to model patterns across variable-length sequences.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["RNN — recurrence through time", "LSTM & GRU — gated memory", "RNN vs Transformer"])

    # ── TAB 1: RNN basics ─────────────────────────────────────────────────
    with tab1:
        st.markdown("### The recurrence equation")
        st.markdown("""
        At each timestep **t**, the RNN takes the current input **xₜ** and the previous
        hidden state **hₜ₋₁**, combines them linearly, and squashes the result through
        an activation function (usually tanh) to produce the new hidden state **hₜ**.
        """)
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.markdown('<div class="formula-box">hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)</div>', unsafe_allow_html=True)
        with col_eq2:
            st.markdown('<div class="formula-box">yₜ = Wᵧ · hₜ + bᵧ<br><small>(optional output at each step)</small></div>', unsafe_allow_html=True)

        st.markdown("### Unrolled RNN — shared weights at every step")

        n_steps = st.slider("Sequence length (timesteps)", 3, 7, 5, key="rnn_steps")
        highlight = st.slider("Highlight timestep", 1, n_steps, 3, key="rnn_hl")

        # Draw unrolled diagram as SVG-style figure with plotly
        fig = go.Figure()
        box_w, box_h = 0.8, 0.5
        y_h, y_x, y_y = 1.5, 0.0, 3.0

        for t in range(n_steps):
            x = t * 2.0
            col_h = "#534AB7" if t+1 == highlight else "#c5c1f0"
            col_x = "#1D9E75" if t+1 == highlight else "#a8dbc9"
            col_y = "#EF9F27" if t+1 == highlight else "#fde3a5"

            # h box
            fig.add_shape(type="rect", x0=x-box_w/2, y0=y_h-box_h/2,
                x1=x+box_w/2, y1=y_h+box_h/2,
                fillcolor=col_h, line=dict(color="#333", width=1.5))
            fig.add_annotation(x=x, y=y_h, text=f"<b>h{t+1}</b>",
                showarrow=False, font=dict(size=12, color="white"))

            # x box
            fig.add_shape(type="rect", x0=x-box_w/2, y0=y_x-box_h/2,
                x1=x+box_w/2, y1=y_x+box_h/2,
                fillcolor=col_x, line=dict(color="#333", width=1))
            fig.add_annotation(x=x, y=y_x, text=f"x{t+1}",
                showarrow=False, font=dict(size=11, color="#1a5c43"))

            # y box
            fig.add_shape(type="rect", x0=x-box_w/2, y0=y_y-box_h/2,
                x1=x+box_w/2, y1=y_y+box_h/2,
                fillcolor=col_y, line=dict(color="#333", width=1))
            fig.add_annotation(x=x, y=y_y, text=f"y{t+1}",
                showarrow=False, font=dict(size=11, color="#7a4e00"))

            # x → h arrow
            fig.add_annotation(x=x, y=y_h-box_h/2, ax=x, ay=y_x+box_h/2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#1D9E75")

            # h → y arrow
            fig.add_annotation(x=x, y=y_y-box_h/2, ax=x, ay=y_h+box_h/2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#EF9F27")

            # h → h+1 arrow
            if t < n_steps - 1:
                fig.add_annotation(x=(t+1)*2.0-box_w/2, y=y_h,
                    ax=x+box_w/2, ay=y_h,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#534AB7")
                fig.add_annotation(x=(x + (t+1)*2.0)/2, y=y_h+0.35,
                    text="Wₕ (shared)", showarrow=False,
                    font=dict(size=9, color="#534AB7"))

        # h0 initial state
        fig.add_shape(type="rect", x0=-2.0-box_w/2, y0=y_h-box_h/2,
            x1=-2.0+box_w/2, y1=y_h+box_h/2,
            fillcolor="#ddd", line=dict(color="#999", width=1.5))
        fig.add_annotation(x=-2.0, y=y_h, text="<b>h₀</b>",
            showarrow=False, font=dict(size=12, color="#555"))
        fig.add_annotation(x=-0.0-box_w/2, y=y_h, ax=-2.0+box_w/2, ay=y_h,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#534AB7")

        # Labels on left
        fig.add_annotation(x=-2.8, y=y_x, text="Inputs xₜ", showarrow=False,
            font=dict(size=11, color="#1D9E75"))
        fig.add_annotation(x=-2.8, y=y_h, text="Hidden hₜ", showarrow=False,
            font=dict(size=11, color="#534AB7"))
        fig.add_annotation(x=-2.8, y=y_y, text="Outputs yₜ", showarrow=False,
            font=dict(size=11, color="#EF9F27"))

        fig.update_layout(
            height=320,
            xaxis=dict(visible=False, range=[-3.5, (n_steps-1)*2.0+1.5]),
            yaxis=dict(visible=False, range=[-0.6, 3.7]),
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Highlighted timestep {highlight}. The same weight matrices Wₕ, Wₓ, Wᵧ are reused at every step — "
                   "this is what 'recurrent' means. h₀ is typically initialised to zeros.")

        st.markdown("""
        **Key properties of vanilla RNNs:**
        - ✅ Handles variable-length sequences naturally
        - ✅ Compact — same weights at every timestep (parameter efficient)
        - ❌ **Vanishing gradient problem** — gradients shrink exponentially as they
          flow back through many timesteps, making it hard to learn long-range dependencies
        - ❌ Sequential computation — each step depends on the previous, so you can't
          parallelise across timesteps the way Transformers can
        """)

    # ── TAB 2: LSTM & GRU ─────────────────────────────────────────────────
    with tab2:
        st.markdown("### Why vanilla RNNs struggle — and how LSTMs fix it")
        st.markdown("""
        When you backpropagate through many timesteps, you multiply gradients together
        repeatedly. If the weights are small, gradients vanish; if large, they explode.
        This makes vanilla RNNs forget events from more than ~10 steps ago.
        """)
        st.markdown('<div class="formula-box">∂Loss/∂h₁ = ∂Loss/∂hₜ · (Wₕ)^(t−1)  →  0 as t grows</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### LSTM — Long Short-Term Memory")
        st.markdown("""
        The LSTM (Hochreiter & Schmidhuber, 1997) adds a **cell state** — a separate
        memory highway — and three **gates** that control what to remember, what to forget,
        and what to output. Gates are sigmoid-activated (0 = fully closed, 1 = fully open).
        """)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**The four LSTM equations:**")
            st.markdown('<div class="formula-box">'
                'fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)   <small>forget gate</small><br>'
                'iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)   <small>input gate</small><br>'
                'c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)  <small>candidate cell</small><br>'
                'oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)   <small>output gate</small>'
                '</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">'
                'cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ   <small>update cell state</small><br>'
                'hₜ = oₜ ⊙ tanh(cₜ)            <small>new hidden state</small>'
                '</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**Interactive gate demo — what each gate does:**")
            forget = st.slider("Forget gate fₜ", 0.0, 1.0, 0.8, step=0.05, key="lstm_f")
            inp    = st.slider("Input gate iₜ",  0.0, 1.0, 0.6, step=0.05, key="lstm_i")
            out    = st.slider("Output gate oₜ", 0.0, 1.0, 0.9, step=0.05, key="lstm_o")
            c_prev = st.slider("Previous cell cₜ₋₁", -2.0, 2.0, 1.0, step=0.1, key="lstm_c")
            c_cand = st.slider("Candidate c̃ₜ (tanh)", -1.0, 1.0, 0.5, step=0.05, key="lstm_cc")

            c_new = forget * c_prev + inp * c_cand
            h_new = out * np.tanh(c_new)

            st.metric("New cell state cₜ", f"{c_new:.3f}",
                delta=f"{c_new - c_prev:.3f} from cₜ₋₁")
            st.metric("New hidden state hₜ", f"{h_new:.3f}")

            if forget < 0.2:
                st.warning("Forget gate ≈ 0 → nearly all previous memory wiped.")
            elif forget > 0.9:
                st.info("Forget gate ≈ 1 → previous cell state passes through almost intact.")
            if inp < 0.1:
                st.warning("Input gate ≈ 0 → new candidate ignored.")
            elif inp > 0.9:
                st.info("Input gate ≈ 1 → new candidate written strongly.")

        st.markdown("---")
        st.markdown("### GRU — Gated Recurrent Unit  *(Cho et al., 2014)*")
        st.markdown("""
        The GRU simplifies the LSTM by merging the cell state and hidden state into one,
        and collapsing the forget and input gates into a single **update gate**. Fewer
        parameters, similar performance, faster to train.
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="formula-box">'
                'zₜ = σ(Wz·[hₜ₋₁, xₜ])   <small>update gate</small><br>'
                'rₜ = σ(Wr·[hₜ₋₁, xₜ])   <small>reset gate</small><br>'
                'h̃ₜ = tanh(W·[rₜ⊙hₜ₋₁, xₜ])  <small>candidate</small><br>'
                'hₜ = (1−zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ  <small>new hidden state</small>'
                '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("""
            - **Update gate zₜ** — blend of old and new (replaces forget + input)
            - **Reset gate rₜ** — how much past hidden state to expose when computing candidate
            - **No separate cell state** — simpler, 33% fewer parameters than LSTM
            - **When to choose:** GRU for speed/simplicity; LSTM when you need finer
              control over what to remember long-term
            """)

    # ── TAB 3: RNN vs Transformer ─────────────────────────────────────────
    with tab3:
        st.markdown("### When do RNNs still make sense?")
        st.markdown("""
        Transformers have largely replaced RNNs for NLP tasks, but RNNs retain real
        advantages in certain settings. Here's an honest comparison.
        """)

        import pandas as pd
        rows = [
            ("Parallelism during training",   "❌ Sequential — step t needs step t−1",   "✅ All tokens processed simultaneously"),
            ("Parallelism during inference",  "✅ Streaming — one step at a time",        "❌ Needs full context window"),
            ("Memory (context length)",       "❌ Degrades for sequences > ~100 steps",  "✅ Scales to thousands of tokens"),
            ("Parameter count",               "✅ Compact for small tasks",              "❌ Typically very large"),
            ("Latency (real-time)",           "✅ Low — constant compute per step",      "❌ Higher — full attention per decode step"),
            ("Long-range dependencies",       "⚠️ LSTMs help but still limited",        "✅ Attention reaches any position directly"),
            ("Time-series (short context)",   "✅ Strong — natural sequential inductive bias", "⚠️ Possible but overkill"),
            ("NLP at scale",                  "❌ Outperformed by Transformers",         "✅ State of the art"),
            ("On-device / edge deployment",   "✅ Small footprint",                      "⚠️ Heavier models harder to deploy"),
        ]
        df = pd.DataFrame(rows, columns=["Property", "RNN / LSTM / GRU", "Transformer"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Architecture family tree")

        timeline = [
            ("1986", "Simple RNN\n(Elman)", "#aaa"),
            ("1997", "LSTM\n(Hochreiter\n& Schmidhuber)", "#534AB7"),
            ("2014", "GRU\n(Cho et al.)", "#534AB7"),
            ("2015", "Bidirectional\nLSTM", "#1D9E75"),
            ("2017", "Transformer\n(Vaswani et al.)", "#E24B4A"),
            ("2023+", "SSMs / Mamba\n(new RNN revival)", "#EF9F27"),
        ]

        fig_tl = go.Figure()
        fig_tl.update_layout(
            height=260,
            xaxis=dict(visible=False, range=[-0.5, len(timeline)-0.5]),
            yaxis=dict(visible=False, range=[-1, 3.2]),
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        for i, (year, name, color) in enumerate(timeline):
            fig_tl.add_shape(type="circle",
                x0=i-0.35, y0=0.6, x1=i+0.35, y1=1.4,
                fillcolor=color, line=dict(color=color))
            fig_tl.add_annotation(x=i, y=1.0, text=f"<b>{year}</b>",
                showarrow=False, font=dict(size=9, color="white"))
            fig_tl.add_annotation(x=i, y=0.15, text=name.replace("\n","<br>"),
                showarrow=False, font=dict(size=9, color="#333"), align="center")
            if i < len(timeline)-1:
                fig_tl.add_shape(type="line", x0=i+0.35, y0=1.0, x1=i+0.65, y1=1.0,
                    line=dict(color="#aaa", width=2))

        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption("Red = Transformer displaced RNNs for most NLP. Orange = modern state-space models "
                   "(Mamba, RWKV) are revisiting recurrence with linear-time attention alternatives.")

        st.markdown("""
        **Bottom line:** if you're building a streaming sensor pipeline, a character-level
        model on a microcontroller, or a small time-series forecaster — RNNs (especially GRUs)
        are still a solid choice. For language at scale, Transformers win. And watch the SSM
        space (Mamba, RWKV) — they combine RNN-style streaming with Transformer-quality results.
        """)

# ═══════════════════════════════════════════════════════════════════════════
# CENTRAL TENDENCY
# ═══════════════════════════════════════════════════════════════════════════
elif section == "central_tendency":
    st.title("📊 Central Tendency")
    st.markdown("""
    <div class="concept-card">
    Measures of <b>central tendency</b> summarise a dataset with a single value that
    represents the centre or most typical value. The choice of measure matters enormously
    — mean, median and mode each tell a different story, and picking the wrong one can
    seriously mislead you.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Mean, median & mode", "Effect of outliers", "When to use which"])

    # ── TAB 1: Definitions & live calculator ─────────────────────────────
    with tab1:
        st.markdown("### The three measures — live on your own data")

        col1, col2 = st.columns([1, 1])
        with col1:
            raw = st.text_area("Enter numbers (comma-separated)",
                value="4, 7, 7, 8, 10, 12, 15, 15, 15, 100",
                height=80, key="ct_raw")
            try:
                data_ct = np.array([float(x.strip()) for x in raw.split(",") if x.strip()])
                valid = True
            except ValueError:
                st.error("Please enter valid numbers separated by commas.")
                data_ct = np.array([4,7,7,8,10,12,15,15,15,100], dtype=float)
                valid = False

            n = len(data_ct)
            mean_val   = np.mean(data_ct)
            median_val = np.median(data_ct)
            # mode: most frequent value(s)
            vals, counts = np.unique(data_ct, return_counts=True)
            max_count = counts.max()
            modes = vals[counts == max_count]
            mode_str = ", ".join([str(int(m) if m == int(m) else m) for m in modes])

            st.markdown(f'<div class="formula-box">'
                f'<b>Mean</b>   x̄ = Σxᵢ / n = {mean_val:.3f}<br>'
                f'<b>Median</b>      = middle value = {median_val:.3f}<br>'
                f'<b>Mode</b>        = most frequent = {mode_str}'
                f'</div>', unsafe_allow_html=True)

            st.markdown(f"n = {n} values &nbsp;|&nbsp; min = {data_ct.min():.2f} &nbsp;|&nbsp; max = {data_ct.max():.2f}")

        with col2:
            fig = go.Figure()
            # Histogram
            fig.add_trace(go.Histogram(x=data_ct, nbinsx=min(20, n),
                marker_color="#c5c1f0", marker_line=dict(color="#534AB7", width=1),
                name="Data", opacity=0.8))
            # Vertical lines for each measure
            for val, label, color in [
                (mean_val,   f"Mean {mean_val:.2f}",   "#E24B4A"),
                (median_val, f"Median {median_val:.2f}", "#1D9E75"),
                (float(modes[0]), f"Mode {mode_str}", "#EF9F27"),
            ]:
                fig.add_vline(x=val, line=dict(color=color, width=2.5, dash="dash"),
                    annotation_text=label, annotation_position="top",
                    annotation_font=dict(color=color, size=11))
            fig.update_layout(height=320, plot_bgcolor="white",
                xaxis_title="Value", yaxis_title="Count",
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=20, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Formulas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Mean (Arithmetic)")
            st.markdown('<div class="formula-box">x̄ = (x₁ + x₂ + … + xₙ) / n<br><br>= Σxᵢ / n</div>', unsafe_allow_html=True)
            st.markdown("Sum all values, divide by count. Sensitive to every data point equally.")
        with col2:
            st.markdown("#### Median")
            st.markdown('<div class="formula-box">Sort the data.<br>If n is odd → middle value<br>If n is even → mean of the two middle values</div>', unsafe_allow_html=True)
            st.markdown("The value that splits the dataset 50/50. Robust to outliers.")
        with col3:
            st.markdown("#### Mode")
            st.markdown('<div class="formula-box">The value (or values) that appear most frequently.<br><br>A dataset can be unimodal, bimodal, or multimodal.</div>', unsafe_allow_html=True)
            st.markdown("The only measure that works for categorical data (e.g. most common colour).")

    # ── TAB 2: Effect of outliers ─────────────────────────────────────────
    with tab2:
        st.markdown("### How outliers pull the mean but not the median")

        col1, col2 = st.columns([1, 2])
        with col1:
            base = st.slider("Typical values (repeated 9×)", 5, 30, 15, key="ct_base")
            outlier = st.slider("Outlier value", int(base), int(base)*20, int(base)*8, key="ct_out")
            include_outlier = st.checkbox("Include outlier", value=True, key="ct_inc")

            dataset = np.array([base]*9 + ([outlier] if include_outlier else []))
            m_mean   = np.mean(dataset)
            m_median = np.median(dataset)

            st.metric("Mean",   f"{m_mean:.1f}", delta=f"{m_mean-base:+.1f} from typical" if include_outlier else None)
            st.metric("Median", f"{m_median:.1f}", delta=f"{m_median-base:+.1f} from typical" if include_outlier else None)

            if include_outlier:
                pull = abs(m_mean - m_median)
                st.markdown(f"The outlier pulls the **mean** {pull:.1f} units away from the median.")
                st.info("Median barely moves — it only cares about rank, not magnitude.")

        with col2:
            fig = go.Figure()
            jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(dataset))
            fig.add_trace(go.Scatter(x=dataset, y=jitter,
                mode="markers", marker=dict(size=14, color="#534AB7", opacity=0.7),
                name="Data points"))
            fig.add_vline(x=m_mean,   line=dict(color="#E24B4A", width=3, dash="dash"),
                annotation_text=f"Mean {m_mean:.1f}", annotation_position="top left",
                annotation_font=dict(color="#E24B4A", size=12))
            fig.add_vline(x=m_median, line=dict(color="#1D9E75", width=3, dash="dot"),
                annotation_text=f"Median {m_median:.1f}", annotation_position="top right",
                annotation_font=dict(color="#1D9E75", size=12))
            fig.update_layout(height=260, plot_bgcolor="white",
                xaxis_title="Value", yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Real-world examples of outlier distortion:**
            - Average household income in a neighbourhood with one billionaire → mean is misleading, median is honest
            - Average response time in a web service with occasional 30-second timeouts → use median or p99
            - Test scores when one student scored 0 due to absence → median better represents the class
            """)

    # ── TAB 3: When to use which ─────────────────────────────────────────
    with tab3:
        st.markdown("### Choosing the right measure")

        import pandas as pd
        rows = [
            ("Symmetric, no outliers (e.g. height)", "✅ Mean — uses all data efficiently", "✅ Median — similar result", "⚠️ Mode — may not be unique"),
            ("Skewed distribution (e.g. income)",    "❌ Distorted by tail",               "✅ Median — robust",         "⚠️ Mode — only shows peak"),
            ("Outliers present",                     "❌ Pulled toward outliers",           "✅ Median — ignores magnitude", "⚠️ Unaffected but limited"),
            ("Categorical data (e.g. colours)",      "❌ Not applicable",                   "❌ Not applicable",          "✅ Mode — only option"),
            ("Reporting 'typical' salary",           "❌ Misleading if skewed",             "✅ Standard practice",       "⚠️ Possible but unusual"),
            ("ML: imputing missing values",          "✅ Common — preserves mean",          "✅ Robust to outliers",      "✅ For categorical features"),
            ("ML: loss function centre",             "✅ MSE minimised at mean",            "✅ MAE minimised at median", "—"),
        ]
        df = pd.DataFrame(rows, columns=["Situation", "Mean", "Median", "Mode"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Skewness and the mean-median relationship")
        col1, col2, col3 = st.columns(3)
        skew_cases = [
            ("Symmetric", "Mean ≈ Median ≈ Mode", "Normal distribution, balanced data", "#534AB7"),
            ("Right-skewed\n(positive)", "Mode < Median < Mean", "Income, house prices, response times — long right tail pulls mean up", "#E24B4A"),
            ("Left-skewed\n(negative)", "Mean < Median < Mode", "Age at retirement, exam scores with ceiling — long left tail pulls mean down", "#1D9E75"),
        ]
        for col, (name, order, example, color) in zip([col1,col2,col3], skew_cases):
            with col:
                st.markdown(f"**{name}**")
                st.markdown(f'<div class="formula-box" style="border-left:4px solid {color}">{order}</div>', unsafe_allow_html=True)
                st.caption(example)

# ═══════════════════════════════════════════════════════════════════════════
# DISPERSION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "dispersion":
    st.title("📏 Dispersion")
    st.markdown("""
    <div class="concept-card">
    Measures of <b>dispersion</b> describe how spread out data is around its centre.
    Two datasets can have identical means yet look completely different — dispersion
    captures that difference. It underpins concepts like variance in the bias-variance
    tradeoff, standard deviation in normalisation, and IQR in outlier detection.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Variance & std dev", "Range, IQR & box plot", "Connecting to ML"])

    # ── TAB 1: Variance & std dev ─────────────────────────────────────────
    with tab1:
        st.markdown("### Variance and standard deviation — average squared deviation from the mean")

        col1, col2 = st.columns([1, 1])
        with col1:
            raw_d = st.text_area("Enter numbers (comma-separated)",
                value="10, 12, 13, 14, 15, 16, 17, 18, 20, 50",
                height=80, key="disp_raw")
            try:
                data_d = np.array([float(x.strip()) for x in raw_d.split(",") if x.strip()])
            except ValueError:
                st.error("Please enter valid numbers.")
                data_d = np.array([10,12,13,14,15,16,17,18,20,50], dtype=float)

            n_d    = len(data_d)
            mean_d = np.mean(data_d)
            # Population vs sample
            pop_var  = np.var(data_d, ddof=0)
            samp_var = np.var(data_d, ddof=1)
            pop_std  = np.std(data_d, ddof=0)
            samp_std = np.std(data_d, ddof=1)

            kind = st.radio("Formula type", ["Population (÷n)", "Sample (÷n−1)"], horizontal=True, key="disp_kind")
            var_show = pop_var  if kind.startswith("Population") else samp_var
            std_show = pop_std  if kind.startswith("Population") else samp_std
            denom    = "n" if kind.startswith("Population") else "n−1"

            st.markdown(f'<div class="formula-box">'
                f'Mean x̄ = {mean_d:.3f}<br><br>'
                f'Variance σ² = Σ(xᵢ − x̄)² / {denom} = {var_show:.3f}<br>'
                f'Std Dev  σ  = √σ² = {std_show:.3f}'
                f'</div>', unsafe_allow_html=True)

            st.caption("Population formula (÷n) for the full dataset. "
                       "Sample formula (÷n−1, Bessel's correction) when estimating from a sample — "
                       "this makes the estimate unbiased.")

        with col2:
            # Show deviations from mean
            deviations = data_d - mean_d
            colors_dev = ["#E24B4A" if d > 0 else "#1D9E75" for d in deviations]

            fig = go.Figure()
            xs = list(range(n_d))
            fig.add_hline(y=mean_d, line=dict(color="#534AB7", width=2, dash="dash"),
                annotation_text=f"Mean={mean_d:.2f}", annotation_position="right")
            fig.add_hline(y=mean_d + std_show, line=dict(color="#EF9F27", width=1.5, dash="dot"),
                annotation_text=f"+1σ={mean_d+std_show:.2f}", annotation_position="right")
            fig.add_hline(y=mean_d - std_show, line=dict(color="#EF9F27", width=1.5, dash="dot"),
                annotation_text=f"−1σ={mean_d-std_show:.2f}", annotation_position="right")

            # Deviation bars
            for i, (x, y, dev, col) in enumerate(zip(xs, data_d, deviations, colors_dev)):
                fig.add_shape(type="line", x0=x, y0=mean_d, x1=x, y1=y,
                    line=dict(color=col, width=2))

            fig.add_trace(go.Scatter(x=xs, y=data_d, mode="markers",
                marker=dict(size=10, color=colors_dev), showlegend=False))

            fig.update_layout(height=320, plot_bgcolor="white",
                xaxis_title="Index", yaxis_title="Value",
                margin=dict(l=20, r=100, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red points are above the mean, green below. "
                       "Vertical lines show each deviation (xᵢ − x̄). "
                       "Orange dashed lines mark ±1 standard deviation.")

        st.markdown("### Breakdown: deviation table")
        import pandas as pd
        dev_table = pd.DataFrame({
            "xᵢ": data_d,
            "xᵢ − x̄": np.round(deviations, 3),
            "(xᵢ − x̄)²": np.round(deviations**2, 3),
        })
        st.dataframe(dev_table, use_container_width=True, hide_index=True, height=200)
        st.markdown(f"Sum of (xᵢ − x̄)² = **{np.sum(deviations**2):.3f}**  "
                    f"÷ {denom} → variance = **{var_show:.3f}** → std dev = **{std_show:.3f}**")

    # ── TAB 2: Range, IQR & box plot ──────────────────────────────────────
    with tab2:
        st.markdown("### Range, IQR and the box plot")

        col1, col2 = st.columns([1, 2])
        with col1:
            raw_b = st.text_area("Enter numbers",
                value="3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 45",
                height=80, key="box_raw")
            try:
                data_b = np.array([float(x.strip()) for x in raw_b.split(",") if x.strip()])
            except ValueError:
                data_b = np.array([3,7,8,9,10,11,12,13,14,15,16,18,45], dtype=float)

            data_b_sorted = np.sort(data_b)
            q1  = np.percentile(data_b, 25)
            q2  = np.percentile(data_b, 50)
            q3  = np.percentile(data_b, 75)
            iqr = q3 - q1
            rng = data_b.max() - data_b.min()
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers_b  = data_b[(data_b < lower_fence) | (data_b > upper_fence)]

            st.markdown('<div class="formula-box">'
                f'Min = {data_b.min():.1f}<br>'
                f'Q1  = {q1:.1f}  (25th percentile)<br>'
                f'Q2  = {q2:.1f}  (median / 50th)<br>'
                f'Q3  = {q3:.1f}  (75th percentile)<br>'
                f'Max = {data_b.max():.1f}<br><br>'
                f'Range = max − min = {rng:.1f}<br>'
                f'IQR   = Q3 − Q1  = {iqr:.1f}'
                '</div>', unsafe_allow_html=True)

            if len(outliers_b) > 0:
                st.warning(f"Outliers (beyond Q1−1.5×IQR or Q3+1.5×IQR): {outliers_b}")
            else:
                st.success("No outliers detected by the 1.5×IQR rule.")

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(y=data_b, boxpoints="all", jitter=0.3,
                marker=dict(color="#534AB7", size=8, opacity=0.7),
                line=dict(color="#534AB7", width=2),
                fillcolor="#c5c1f0",
                name="Data"))

            # Annotations for quartiles
            for val, label in [(q1,"Q1"),(q2,"Q2 (median)"),(q3,"Q3")]:
                fig.add_annotation(x=0.45, y=val, text=f"  {label}={val:.1f}",
                    showarrow=False, font=dict(size=11, color="#534AB7"),
                    xref="paper")

            fig.update_layout(height=380, plot_bgcolor="white",
                yaxis_title="Value", showlegend=False,
                margin=dict(l=20, r=120, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Reading the box plot:**
        - **Box** spans Q1 to Q3 — contains the middle 50% of data (the IQR)
        - **Line inside box** = median (Q2)
        - **Whiskers** extend to the last point within 1.5×IQR of the box
        - **Dots beyond whiskers** = outliers by Tukey's rule
        - **Wide box** = high spread; **narrow box** = concentrated data

        **Range vs IQR:**
        | | Range | IQR |
        |---|---|---|
        | Formula | max − min | Q3 − Q1 |
        | Sensitive to outliers | ❌ Yes — heavily | ✅ No — ignores top/bottom 25% |
        | Use when | Quick rough spread | Robust spread for skewed data |
        """)

    # ── TAB 3: Connecting to ML ───────────────────────────────────────────
    with tab3:
        st.markdown("### Why dispersion is central to ML")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature normalisation")
            st.markdown("""
            Most ML models are sensitive to the scale of features. Dispersion measures
            define the two most common normalisation strategies:
            """)
            st.markdown('<div class="formula-box">'
                '<b>Z-score (standardisation)</b><br>'
                'x′ = (x − μ) / σ<br><br>'
                '<b>Min-max scaling</b><br>'
                'x′ = (x − min) / (max − min)<br><br>'
                '<b>Robust scaling</b><br>'
                'x′ = (x − Q2) / IQR'
                '</div>', unsafe_allow_html=True)
            st.caption("Robust scaling uses median and IQR — best when outliers are present.")

        with col2:
            st.markdown("#### Bias-variance tradeoff")
            st.markdown("""
            The **variance** of a model's predictions across different training sets is
            literally the statistical variance you saw in Tab 1 — it measures how much
            the model's output fluctuates when trained on different data samples.
            """)
            st.markdown('<div class="formula-box">'
                'Expected Error =<br>'
                'Bias² + Variance + Irreducible Noise'
                '</div>', unsafe_allow_html=True)
            st.caption("High-variance models (deep trees, large nets) overfit — "
                       "their predictions have high spread across training sets.")

        st.markdown("---")
        st.markdown("### Interactive: same mean, different spread")
        st.markdown("Two datasets can have identical central tendency but very different dispersion:")

        spread_a = st.slider("Spread of Dataset A (std dev)", 0.5, 10.0, 1.5, step=0.5, key="disp_a")
        spread_b = st.slider("Spread of Dataset B (std dev)", 0.5, 10.0, 6.0, step=0.5, key="disp_b")

        rng_d = np.random.default_rng(42)
        da = rng_d.normal(loc=10, scale=spread_a, size=200)
        db = rng_d.normal(loc=10, scale=spread_b, size=200)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=da, nbinsx=30, name=f"A  σ={spread_a}",
            marker_color="#534AB7", opacity=0.6))
        fig.add_trace(go.Histogram(x=db, nbinsx=30, name=f"B  σ={spread_b}",
            marker_color="#E24B4A", opacity=0.6))
        fig.add_vline(x=10, line=dict(color="#333", width=2, dash="dash"),
            annotation_text="Mean=10", annotation_position="top")
        fig.update_layout(barmode="overlay", height=280, plot_bgcolor="white",
            xaxis_title="Value", yaxis_title="Count",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Both distributions have mean = 10. A model predicting only the mean "
                   "would be equally wrong about both — yet B's errors are far more severe.")

        import pandas as pd
        summary_df = pd.DataFrame({
            "Measure": ["Mean", "Std Dev", "Variance", "Range (approx)"],
            "Dataset A": [f"{np.mean(da):.2f}", f"{np.std(da):.2f}", f"{np.var(da):.2f}", f"{da.max()-da.min():.2f}"],
            "Dataset B": [f"{np.mean(db):.2f}", f"{np.std(db):.2f}", f"{np.var(db):.2f}", f"{db.max()-db.min():.2f}"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
