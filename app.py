import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from knn_models import BruteForceKNN, KDTreeKNN

st.set_page_config(
    page_title="DSA Project",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    div[data-testid="stMetricValue"] {
        color: #10B981 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title(" Forest Covertype: KNN VS BRUTE-FORCE")
st.write(
    "Real-time Brute-force KNN vs KD-Tree KNN using the 10 original continuous terrain features ($D \\le 10$)."
)

# Target Class Label Mapping
COVERTYPE_CLASS_MAP = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

# The 10 Continuous Terrain Attributes
CONTINUOUS_10_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

@st.cache_data
def load_and_preprocess_covertype_fast():
    # Loads 10 continuous features into C-contiguous memory arrays for instant execution.
    CACHE_FILE = "covertype_10d_raw.csv"

    if os.path.exists(CACHE_FILE):
        data = pd.read_csv(CACHE_FILE)
    else:
        covertype = fetch_ucirepo(id=31)
        X = covertype.data.features[CONTINUOUS_10_COLS].copy()
        y = covertype.data.targets.copy()

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        data = X.copy()
        data["Cover_Type"] = y.to_numpy()

        # Clean missing values
        data = data.dropna().reset_index(drop=True)
        numeric_cols = data.select_dtypes(include=np.number).columns
        data = data[np.isfinite(data[numeric_cols]).all(axis=1)].reset_index(drop=True)

        # Sample 100,000 rows max for optimal RAM speed
        if len(data) > 100000:
            data, _ = train_test_split(
                data, train_size=100000, random_state=42, stratify=data["Cover_Type"]
            )
            data = data.reset_index(drop=True)

        data.to_csv(CACHE_FILE, index=False)

    X_clean = data[CONTINUOUS_10_COLS].astype(np.float64)
    y_clean = data["Cover_Type"].astype(np.int64)

    # Standardize all 10 continuous features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 80/20 Stratified Train/Test Split
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_scaled, y_clean, test_size=0.20, random_state=42, stratify=y_clean
    )

    return (
        np.ascontiguousarray(X_train_full, dtype=np.float64),
        np.ascontiguousarray(X_test_full, dtype=np.float64),
        y_train.to_numpy(),
        y_test.to_numpy(),
        scaler,
        CONTINUOUS_10_COLS,
    )


# Model Caching to Avoid Tree Re-building on Slider Changes
@st.cache_resource
def get_cached_kd_tree(X_train_sub, y_train_sub, k_val, leaf_val):
    #Builds and caches the KD-Tree in RAM. Prevents lag when dragging UI sliders.
    tree = KDTreeKNN(k=k_val, leaf_size=leaf_val)
    tree.fit(X_train_sub, y_train_sub)
    return tree



st.sidebar.header("Parameters")

n_dimensions = st.sidebar.slider(
    "Continuous Features Count (D)",
    min_value=2,
    max_value=10,
    value=5,
    step=1,
    help="Select how many continuous physical terrain dimensions (2 to 10) to include.",
)

try:
    (
        X_train_full,
        X_test_full,
        y_train,
        y_test,
        scaler,
        all_feature_cols,
    ) = load_and_preprocess_covertype_fast()

    # Instant slicing
    selected_feature_cols = all_feature_cols[:n_dimensions]
    X_train = np.ascontiguousarray(X_train_full[:, :n_dimensions])
    X_test = np.ascontiguousarray(X_test_full[:, :n_dimensions])

    st.sidebar.success(
        f"Pipeline Ready ({len(X_train):,} Train rows, D={n_dimensions})"
    )
except Exception as e:
    st.error(f"Error executing data pipeline: {e}")
    st.stop()

max_subset = st.sidebar.slider(
    "Training Set Size (N)",
    min_value=5000,
    max_value=min(80000, len(X_train)),
    value=40000,
    step=5000,
)

leaf_bucket_size = st.sidebar.slider(
    "Vectorized Leaf Bucket Size",
    min_value=8,
    max_value=64,
    value=32,
    step=8,
)

k_neighbors = st.sidebar.slider(
    "Number of Neighbors (k)", min_value=1, max_value=15, value=5
)

tab1, tab2 = st.tabs(
    ["Speed Comparison", "Terrain Classifier & Physical Recommender"]
)

with tab1:
    st.subheader(f"Speed Benchmark ({n_dimensions} Continuous Features)")
    st.write(
        "Evaluating the performance of KNN implemented through brute force and KD-Tree across increasing dataset sizes ($N$) and feature dimensions ($D$)."
    )

    n_queries = st.slider(
        "Test Query Batch Size (Q)",
        min_value=10,
        max_value=min(200, len(X_test)),
        value=50,
        step=10,
    )

    if st.button("Compare", type="primary"):
        # Fast 5-point evaluation grid
        sample_sizes = np.linspace(5000, max_subset, 5, dtype=int).tolist()

        bf_query_times = []
        kd_build_times = []
        kd_query_times = []
        accuracies_kd = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        X_test_eval = np.ascontiguousarray(X_test[:n_queries])
        y_test_eval = y_test[:n_queries]

        for idx, N in enumerate(sample_sizes):
            status_text.text(f"Benchmarking N = {N:,} samples...")

            X_train_eval = np.ascontiguousarray(X_train[:N])
            y_train_eval = y_train[:N]

            # Brute Force Benchmark
            bf_model = BruteForceKNN(k=k_neighbors)
            bf_model.fit(X_train_eval, y_train_eval)

            t0 = time.perf_counter()
            _ = bf_model.predict(X_test_eval)
            bf_query_times.append((time.perf_counter() - t0) * 1000)

            # KD-Tree Benchmark
            t0 = time.perf_counter()
            kd_model = KDTreeKNN(k=k_neighbors, leaf_size=leaf_bucket_size)
            kd_model.fit(X_train_eval, y_train_eval)
            kd_build_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            kd_preds = kd_model.predict(X_test_eval)
            kd_query_times.append((time.perf_counter() - t0) * 1000)

            accuracies_kd.append(float(np.mean(kd_preds == y_test_eval)) * 100)
            progress_bar.progress((idx + 1) / len(sample_sizes))

        status_text.success("Successfully completed the commparison between Brute-Force and KD-Tree KNN algorithms.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Brute-Force Query", f"{np.mean(bf_query_times):.2f} ms")
        with c2:
            speedup = (1 - np.mean(kd_query_times) / np.mean(bf_query_times)) * 100
            st.metric(
                "Avg KD-Tree Query",
                f"{np.mean(kd_query_times):.2f} ms",
                delta=f"{speedup:.1f}% Faster" if speedup >= 0 else f"-{abs(speedup):.1f}% Slower",
            )
        with c3:
            st.metric("Avg Tree Build Time", f"{np.mean(kd_build_times):.2f} ms")
        with c4:
            st.metric("Test Accuracy (k-NN)", f"{accuracies_kd[-1]:.2f}%")

        st.markdown("---")

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F172A")

        ax1.plot(sample_sizes, bf_query_times, marker="o", linewidth=2.5, color="#8B5CF6", label="Brute Force O(N·D)")
        ax1.plot(sample_sizes, kd_query_times, marker="s", linewidth=2.5, color="#10B981", label=f"KD-Tree O(D log N) [{n_dimensions}D]")
        ax1.set_title(f"Query Latency ({n_queries} test queries)", color="#F8FAFC")
        ax1.set_xlabel("Training Set Size (N)", color="#CBD5E1")
        ax1.set_ylabel("Time (ms)", color="#CBD5E1")
        ax1.grid(True, linestyle="--", alpha=0.2)
        ax1.legend(facecolor="#1E293B", edgecolor="#334155")
        ax1.set_facecolor("#0F172A")

        ax2.plot(sample_sizes, kd_build_times, marker="^", linewidth=2.5, color="#F59E0B", label="KD-Tree Build Overhead")
        ax2.set_title("Tree Construction Overhead", color="#F8FAFC")
        ax2.set_xlabel("Training Set Size (N)", color="#CBD5E1")
        ax2.set_ylabel("Time (ms)", color="#CBD5E1")
        ax2.grid(True, linestyle="--", alpha=0.2)
        ax2.legend(facecolor="#1E293B", edgecolor="#334155")
        ax2.set_facecolor("#0F172A")

        st.pyplot(fig)
    else:
        st.info("Click **Compare** to compare the performance of different algorithms.")


with tab2:
    st.subheader(f"Real-Time Terrain Match Recommender ({n_dimensions}D)")
    st.write("Drag feature sliders to query the KD-Tree.")

    X_train_sub = np.ascontiguousarray(X_train[:max_subset])
    y_train_sub = y_train[:max_subset]

    # Fetch cached pre-built tree (0 ms rebuild delay)
    kd_engine = get_cached_kd_tree(X_train_sub, y_train_sub, k_neighbors, leaf_bucket_size)

    # Dynamic Feature Input Sliders
    feat_values = []
    cols = st.columns(min(n_dimensions, 4))
    for i in range(n_dimensions):
        col_idx = i % len(cols)
        min_v = float(X_train[:, i].min())
        max_v = float(X_train[:, i].max())
        val = cols[col_idx].slider(
            f"{selected_feature_cols[i]} (scaled)",
            min_value=min_v,
            max_value=max_v,
            value=0.0,
            step=0.1,
        )
        feat_values.append(val)

    query_vec = np.array(feat_values, dtype=np.float64)

    t0 = time.perf_counter()
    distances, indices = kd_engine.query(query_vec, k=k_neighbors)
    query_latency = (time.perf_counter() - t0) * 1000

    pred_class_id = kd_engine.predict_one(query_vec)
    pred_class_name = COVERTYPE_CLASS_MAP.get(pred_class_id, f"Type {pred_class_id}")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Predicted Forest Cover Type", pred_class_name)
    with res_col2:
        st.metric("KD-Tree Prediction Time", f"{query_latency:.3f} ms")

    st.markdown("###Top-K Recommended Neighbor Matches)")

    recommendations = []
    for rank, (dist, idx) in enumerate(zip(distances, indices), start=1):
        scaled_row = X_train_full[idx]
        raw_coords = scaler.inverse_transform([scaled_row])[0]

        cat_id = y_train_sub[idx]
        cat_name = COVERTYPE_CLASS_MAP.get(cat_id, f"Type {cat_id}")

        rec_dict = {
            "Rank": rank,
            "Dataset Index": idx,
            "Euclidean Distance": round(dist, 4),
            "Cover Type": cat_name,
        }

        for feat_idx, feat_name in enumerate(selected_feature_cols):
            rec_dict[feat_name] = round(raw_coords[feat_idx], 2)

        recommendations.append(rec_dict)

    df_recs = pd.DataFrame(recommendations)
    st.dataframe(df_recs, use_container_width=True, hide_index=True)