import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ML Концепции — Интерактивно",
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

# ── Sidebar navigation ──────────────────────────────────────────────────────

SECTIONS = {
    "🏠 Начало": "home",
    "📉 Loss функция": "loss",
    "🏔️ Градиент & Descent": "gradient",
    "🔄 Backpropagation": "backprop",
    "⚖️ Overfitting / Underfitting": "overfit",
    "🎯 Активационни функции": "activation",
    "📊 Линейна регресия": "linear_reg",
}

with st.sidebar:
    st.title("🧠 ML Концепции")
    st.caption("Интерактивно ръководство")
    st.divider()
    selection = st.radio("Раздел", list(SECTIONS.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Всичко работи локално — без интернет")

section = SECTIONS[selection]

# ═══════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════
if section == "home":
    st.title("Интерактивно ръководство по машинно обучение")
    st.markdown("Избери раздел от менюто вляво, за да разгледаш концепция интерактивно.")
    st.divider()

    cols = st.columns(3)
    cards = [
        ("📉", "Loss функция", "Как измерваме грешката на модела"),
        ("🏔️", "Градиент & Descent", "Как моделът се учи стъпка по стъпка"),
        ("🔄", "Backpropagation", "Как грешката се разпространява назад"),
        ("⚖️", "Overfitting", "Твърде много или твърде малко обучение"),
        ("🎯", "Активационни функции", "ReLU, Sigmoid, Tanh и техните свойства"),
        ("📊", "Линейна регресия", "Намиране на права линия през данните"),
    ]
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="border:0.5px solid #d3d1c7;border-radius:12px;
                        padding:1.2rem;margin-bottom:1rem;min-height:110px">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:500;margin:6px 0 4px">{title}</div>
                <div style="font-size:0.85rem;color:#5F5E5A">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "loss":
    st.title("📉 Loss функция")
    st.markdown("""
    <div class="concept-card">
    <b>Loss функцията</b> мери колко грешен е моделът. Тя взима предсказанието и
    истинската стойност и връща едно число — <em>колко зле се справяме</em>.
    Целта на обучението е да минимизираме тази стойност.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["MSE — Средна квадратична грешка", "MAE — Средна абсолютна грешка", "Binary Cross-Entropy"])

    with tab1:
        st.markdown('<div class="formula-box">MSE = (1/n) · Σ (y_true − y_pred)²</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Параметри**")
            n_pts = st.slider("Брой точки", 5, 30, 10, key="mse_n")
            noise = st.slider("Шум в данните", 0.1, 3.0, 1.0, key="mse_noise")
            pred_offset = st.slider("Отместване на предсказанието", -3.0, 3.0, 1.0, key="mse_off", step=0.1)
            np.random.seed(42)
            x_data = np.linspace(0, 10, n_pts)
            y_true = 2 * x_data + 1 + np.random.normal(0, noise, n_pts)
            y_pred = 2 * x_data + 1 + pred_offset
            mse_val = np.mean((y_true - y_pred) ** 2)
            mae_val = np.mean(np.abs(y_true - y_pred))
            st.metric("MSE", f"{mse_val:.3f}")
            st.metric("MAE", f"{mae_val:.3f}")
            st.info("MSE наказва по-силно големите грешки заради квадрата.")
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers',
                name='Истински стойности', marker=dict(color='#534AB7', size=8)))
            fig.add_trace(go.Scatter(x=x_data, y=y_pred, mode='lines',
                name='Предсказание', line=dict(color='#E24B4A', width=2)))
            for xi, yt, yp in zip(x_data, y_true, y_pred):
                fig.add_shape(type='line', x0=xi, x1=xi, y0=yt, y1=yp,
                    line=dict(color='#EF9F27', width=1.5, dash='dot'))
            fig.update_layout(title="Данни, предсказание и грешки (оранжево)",
                xaxis_title="x", yaxis_title="y", height=380,
                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="formula-box">MAE = (1/n) · Σ |y_true − y_pred|</div>', unsafe_allow_html=True)
        st.markdown("MAE и MSE сравнени за различни размери грешка:")
        errors = np.linspace(0, 4, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=errors, y=errors**2, name='MSE грешка (e²)',
            line=dict(color='#534AB7', width=2.5)))
        fig.add_trace(go.Scatter(x=errors, y=errors, name='MAE грешка (|e|)',
            line=dict(color='#1D9E75', width=2.5)))
        fig.update_layout(xaxis_title="Грешка e", yaxis_title="Принос към loss",
            height=360, legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        - **MSE** расте квадратично → силно наказва outliers
        - **MAE** расте линейно → по-устойчив на outliers
        """)

    with tab3:
        st.markdown('<div class="formula-box">BCE = −[y·log(p) + (1−y)·log(1−p)]</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            true_label = st.radio("Истински клас (y)", [1, 0], key="bce_y")
            pred_prob = st.slider("Предсказана вероятност (p)", 0.01, 0.99, 0.7, step=0.01, key="bce_p")
            bce = -(true_label * np.log(pred_prob) + (1 - true_label) * np.log(1 - pred_prob))
            st.metric("BCE Loss", f"{bce:.4f}")
            if bce < 0.3:
                st.success("Много добро предсказание!")
            elif bce < 1.0:
                st.warning("Умерена грешка")
            else:
                st.error("Голяма грешка!")
        with col2:
            probs = np.linspace(0.01, 0.99, 300)
            loss_y1 = -np.log(probs)
            loss_y0 = -np.log(1 - probs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=probs, y=loss_y1, name='y=1 (трябва p→1)',
                line=dict(color='#534AB7', width=2.5)))
            fig.add_trace(go.Scatter(x=probs, y=loss_y0, name='y=0 (трябва p→0)',
                line=dict(color='#D85A30', width=2.5)))
            fig.add_vline(x=pred_prob, line_dash='dash', line_color='#EF9F27',
                annotation_text=f"p={pred_prob:.2f}")
            fig.update_layout(xaxis_title="Предсказана вероятност",
                yaxis_title="Loss", yaxis_range=[0, 5], height=360,
                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# GRADIENT DESCENT
# ═══════════════════════════════════════════════════════════════════════════
elif section == "gradient":
    st.title("🏔️ Градиент и Gradient Descent")
    st.markdown("""
    <div class="concept-card">
    <b>Градиентът</b> е вектор, който показва <em>в каква посока функцията расте най-бързо</em>.
    За да минимизираме loss, вървим в <em>обратната</em> посока — това е <b>gradient descent</b>.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["1D симулация", "2D Loss повърхност"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Настройки**")
            lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5], value=0.1)
            start_x = st.slider("Начална точка x₀", -4.0, 4.0, 3.0, step=0.1)
            n_steps = st.slider("Брой стъпки", 5, 60, 25)
            loss_type = st.selectbox("Loss форма", ["Парабола x²", "Асиметрична", "С локален минимум"])

        def get_loss_and_grad(loss_type):
            if loss_type == "Парабола x²":
                return lambda x: x**2, lambda x: 2*x
            elif loss_type == "Асиметрична":
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
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name='Loss крива',
                line=dict(color='#AFA9EC', width=2.5)))
            fig.add_trace(go.Scatter(
                x=xs_hist, y=ys_hist, mode='markers+lines',
                name='Gradient descent стъпки',
                marker=dict(color=list(range(len(xs_hist))), colorscale='RdYlGn_r',
                    size=9, showscale=True, colorbar=dict(title='Стъпка', thickness=12)),
                line=dict(color='rgba(239,159,39,0.4)', width=1.5, dash='dot')
            ))
            fig.add_scatter(x=[xs_hist[0]], y=[ys_hist[0]], mode='markers',
                marker=dict(color='#E24B4A', size=14, symbol='circle'),
                name='Начало', showlegend=True)
            fig.add_scatter(x=[xs_hist[-1]], y=[ys_hist[-1]], mode='markers',
                marker=dict(color='#1D9E75', size=14, symbol='star'),
                name='Край', showlegend=True)
            fig.update_layout(xaxis_title="x (параметър)", yaxis_title="Loss",
                height=400, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Начален x", f"{xs_hist[0]:.3f}")
        c2.metric("Краен x", f"{xs_hist[-1]:.3f}")
        c3.metric("Стъпки до конвергенция", len(xs_hist) - 1)

        if lr >= 1.0:
            st.warning("Голям learning rate — моделът може да 'прескача' минимума и да се дестабилизира.")
        elif lr <= 0.05:
            st.info("Малък learning rate — конвергенцията е стабилна, но бавна.")
        else:
            st.success("Добър learning rate — бързо и стабилно спускане.")

    with tab2:
        st.markdown("Loss повърхност с два параметъра (w₁, w₂):")
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
        st.caption("Завърти с мишката. Целта е да намерим долината — минималния Loss.")

# ═══════════════════════════════════════════════════════════════════════════
# BACKPROPAGATION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "backprop":
    st.title("🔄 Backpropagation")
    st.markdown("""
    <div class="concept-card">
    <b>Backpropagation</b> (обратно разпространение) е алгоритъмът, с който изчисляваме
    градиента на всеки параметър в мрежата. Използва <em>верижното правило</em> от математическия анализ,
    за да пренесе грешката от изхода назад към входа.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Ръчен пример — проста мрежа")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Входни данни**")
        x_in = st.slider("Вход x", 0.1, 3.0, 1.0, step=0.1)
        w1_bp = st.slider("Тегло w₁", -3.0, 3.0, 0.5, step=0.1)
        w2_bp = st.slider("Тегло w₂", -3.0, 3.0, 2.0, step=0.1)
        y_target = st.slider("Целева стойност y", 0.0, 5.0, 2.0, step=0.1)

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
h        = x · w₁ = {h:.4f}
σ(h)     = {sigmoid_h:.4f}
output   = σ(h) · w₂ = {output:.4f}
loss     = ½(output - y)² = {loss_bp:.4f}
""", language="text")
        st.markdown("**Backward pass (градиенти)**")
        st.code(f"""
∂loss/∂w₂ = {grad_w2:.4f}
∂loss/∂w₁ = {grad_w1:.4f}

(верижно правило:)
∂loss/∂w₁ = (output-y) · w₂ · σ'(h) · x
""", language="text")

    st.markdown("### Как се променят теглата след стъпка")
    lr_bp = st.slider("Learning rate", 0.01, 1.0, 0.1, step=0.01)
    w1_new = w1_bp - lr_bp * grad_w1
    w2_new = w2_bp - lr_bp * grad_w2

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

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss при обучение", "Промяна на теглата"])
    fig.add_trace(go.Scatter(y=loss_hist, name='Loss', line=dict(color='#E24B4A', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=w1_hist, name='w₁', line=dict(color='#534AB7', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=w2_hist, name='w₂', line=dict(color='#1D9E75', width=2)), row=1, col=2)
    fig.update_xaxes(title_text="Стъпка")
    fig.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# OVERFITTING / UNDERFITTING
# ═══════════════════════════════════════════════════════════════════════════
elif section == "overfit":
    st.title("⚖️ Overfitting и Underfitting")
    st.markdown("""
    <div class="concept-card">
    <b>Underfitting</b> — моделът е прекалено прост и не улавя структурата в данните.<br>
    <b>Overfitting</b> — моделът се е „научил наизуст" на тренировъчните данни
    и се справя зле с нови примери.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        degree = st.slider("Степен на полинома", 1, 15, 3)
        noise_of = st.slider("Шум", 0.1, 1.5, 0.5, step=0.1)
        n_train = st.slider("Тренировъчни точки", 8, 25, 12)

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
            name='Тренировъчни', marker=dict(color='#534AB7', size=9)))
        fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers',
            name='Тестови', marker=dict(color='#1D9E75', size=7, symbol='x')))
        fig.add_trace(go.Scatter(x=x_line, y=np.sin(x_line),
            name='Истинска функция', line=dict(color='#888780', dash='dash', width=1.5)))
        clip = np.clip(y_line, -4, 4)
        fig.add_trace(go.Scatter(x=x_line, y=clip,
            name=f'Полином (степен {degree})', line=dict(color='#E24B4A', width=2.5)))
        fig.update_layout(xaxis_title="x", yaxis_title="y",
            yaxis_range=[-3, 3], height=380, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train MSE", f"{train_mse:.4f}")
    c2.metric("Test MSE", f"{test_mse:.4f}")
    ratio = test_mse / (train_mse + 1e-9)
    c3.metric("Test/Train ratio", f"{ratio:.2f}")

    if degree <= 2:
        st.warning("**Underfitting** — моделът е прекалено прост, не улавя синусоидата.")
    elif degree >= 9:
        st.error('**Overfitting** — моделът се извива прекалено и "запаметява" шума.')
    else:
        st.success("**Добър баланс** — моделът улавя структурата без да запаметява шума.")

    st.markdown("### Как MSE зависи от сложността на модела")
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
        annotation_text=f"Избрана степен: {degree}")
    fig2.update_layout(xaxis_title="Степен на полинома",
        yaxis_title="MSE", yaxis_range=[0, 5], height=300,
        legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "activation":
    st.title("🎯 Активационни функции")
    st.markdown("""
    <div class="concept-card">
    Активационните функции добавят <b>нелинейност</b> към невронната мрежа.
    Без тях, колкото и слоя да имаме, мрежата би могла да научи само линейни трансформации.
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

    selected = st.multiselect("Функции за сравнение", list(funcs.keys()),
        default=["ReLU", "Sigmoid", "Tanh"])

    x_act = np.linspace(-4, 4, 300)
    colors = ['#534AB7', '#E24B4A', '#1D9E75', '#EF9F27', '#D4537E']

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Функция f(x)", "Производна f'(x)"])
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

    st.markdown("### Свойства")
    props = {
        "ReLU":       ("Прост, бърз", "Dead neurons при x<0", "Скрити слоеве"),
        "Sigmoid":    ("Изход [0,1]", "Vanishing gradient", "Класификация (изход)"),
        "Tanh":       ("Центриран около 0", "Vanishing gradient", "Скрити слоеве"),
        "Leaky ReLU": ("Без dead neurons", "Допълнителен хиперпараметър", "Скрити слоеве"),
        "ELU":        ("Гладък, по-бърза конвергенция", "По-бавно изчисляване", "Скрити слоеве"),
    }
    rows = [(n,) + props[n] for n in selected if n in props]
    if rows:
        import pandas as pd
        df_props = pd.DataFrame(rows, columns=["Функция", "Предимство", "Недостатък", "Приложение"])
        st.dataframe(df_props, use_container_width=True, hide_index=True)

    st.markdown("### Интерактивен пример — ефектът на активацията")
    x_demo = st.slider("Вход x", -4.0, 4.0, 1.5, step=0.1)
    chosen_fn = st.selectbox("Избери функция", list(funcs.keys()))
    f_demo, df_demo = funcs[chosen_fn]
    out_demo = f_demo(np.array([x_demo]))[0]
    grad_demo = df_demo(np.array([x_demo]))[0]
    d1, d2 = st.columns(2)
    d1.metric(f"{chosen_fn}({x_demo:.1f})", f"{out_demo:.4f}")
    d2.metric(f"Градиент при x={x_demo:.1f}", f"{grad_demo:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif section == "linear_reg":
    st.title("📊 Линейна регресия")
    st.markdown("""
    <div class="concept-card">
    Линейната регресия търси права линия <b>y = w·x + b</b>, която минимизира MSE
    между предсказаните и истинските стойности. Теглото <code>w</code> и отместването
    <code>b</code> се намират чрез gradient descent или аналитично решение.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Интерактивна регресия", "Gradient descent по w и b"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            true_w = st.slider("Истинско w", -3.0, 3.0, 1.5, step=0.1, key="lr_w")
            true_b = st.slider("Истинско b", -3.0, 3.0, 0.5, step=0.1, key="lr_b")
            noise_lr = st.slider("Шум", 0.1, 3.0, 1.0, step=0.1, key="lr_n")
            n_lr = st.slider("Брой точки", 10, 60, 25, key="lr_pts")

        np.random.seed(42)
        x_lr = np.random.uniform(-4, 4, n_lr)
        y_lr = true_w * x_lr + true_b + np.random.normal(0, noise_lr, n_lr)

        X_mat = np.column_stack([x_lr, np.ones(n_lr)])
        params = np.linalg.lstsq(X_mat, y_lr, rcond=None)[0]
        w_hat, b_hat = params

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_lr, y=y_lr, mode='markers',
                name='Данни', marker=dict(color='#534AB7', size=7)))
            x_fit = np.linspace(-4.5, 4.5, 100)
            fig.add_trace(go.Scatter(x=x_fit, y=w_hat*x_fit + b_hat,
                name=f'Регресия: y={w_hat:.2f}x + {b_hat:.2f}',
                line=dict(color='#E24B4A', width=2.5)))
            fig.add_trace(go.Scatter(x=x_fit, y=true_w*x_fit + true_b,
                name=f'Истинска: y={true_w}x + {true_b}',
                line=dict(color='#1D9E75', width=1.5, dash='dash')))
            fig.update_layout(xaxis_title="x", yaxis_title="y",
                height=360, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Намерено w", f"{w_hat:.3f}", f"{w_hat - true_w:+.3f}")
        c2.metric("Намерено b", f"{b_hat:.3f}", f"{b_hat - true_b:+.3f}")
        c3.metric("Train MSE", f"{np.mean((w_hat*x_lr+b_hat - y_lr)**2):.3f}")
        c4.metric("R²", f"{1 - np.var(y_lr - (w_hat*x_lr+b_hat))/np.var(y_lr):.4f}")

    with tab2:
        st.markdown("Loss повърхност по параметрите **w** и **b**:")
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
            name='Намерен минимум')
        fig3.update_layout(xaxis_title='w', yaxis_title='b',
            height=400, legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Зелената звезда е минимумът на MSE — оптималните w и b.")
