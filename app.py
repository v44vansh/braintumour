"""
Brain Tumour Segmentation (U-Net, multimodal MRI) - Streamlit app
Run:  streamlit run app.py
Educational project only - not a medical device.
"""
import os, tempfile, time
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ------------------------------------------------------------------ config
IMG = 128
MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]          # same order as the notebook
CLASS_NAMES = ["Background", "Necrotic core", "Oedema", "Enhancing tumour"]
COLORS = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}   # red / green / yellow
REGIONS = {
    "WT (whole tumour)":     lambda a: a > 0,
    "TC (tumour core)":      lambda a: (a == 1) | (a == 3),
    "ET (enhancing tumour)": lambda a: a == 3,
}

st.set_page_config(page_title="NeuroScan U-Net", page_icon="🧠", layout="wide")


# ------------------------------------------------------------------ notebook functions (unchanged)
def normalize_slice(s):
    s = np.asarray(s, dtype=np.float32)
    mask = s > 0.01 * (s.max() + 1e-8)
    if mask.sum() < 50:
        return np.zeros_like(s)
    lo, hi = np.percentile(s[mask], [1, 99.5])
    out = np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)
    out[~mask] = 0
    return out


def detect_modality(text):
    t = text.lower().replace("-", "_")
    if "t1ce" in t or "t1c" in t or "t1gd" in t or "t1_ce" in t or "t1_contrast" in t:
        return "T1ce"
    if "flair" in t:
        return "FLAIR"
    if "t2" in t:
        return "T2"
    if "t1" in t:
        return "T1"
    return None


def prep_slice(sl):
    return cv2.resize(normalize_slice(sl), (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_input(slices):
    x = np.zeros((1, IMG, IMG, 4), np.float32)
    for m, sl in slices.items():
        x[0, ..., MODALITIES.index(m)] = prep_slice(sl)
    return x


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def read_upload(f):
    """Return (volume HxWxD float32, voxel spacing (3,))"""
    name = f.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(f.getvalue())
            p = tmp.name
        img = nib.load(p)
        vol = img.get_fdata(dtype=np.float32)
        vol = vol[..., 0] if vol.ndim == 4 else vol
        zooms = np.array(img.header.get_zooms()[:3], dtype=float)
        os.remove(p)
        return vol, zooms
    arr = np.array(Image.open(f).convert("L"), dtype=np.float32)   # png / jpg slice
    return arr[..., None], np.array([1.0, 1.0, 1.0])


def rot(v, z):
    return np.rot90(v[:, :, z])


def predict_volume(model, vols):
    depth = next(iter(vols.values())).shape[2]
    batch = np.concatenate(
        [build_input({m: rot(v, z) for m, v in vols.items()}) for z in range(depth)]
    )
    t0 = time.time()
    pred = model.predict(batch, batch_size=32, verbose=0)
    return pred.argmax(-1).astype(np.uint8), pred.max(-1), time.time() - t0


def to_rgb(gray):
    g = normalize_slice(gray)
    return np.stack([g] * 3, -1)


def overlay(gray, mask128, alpha, show):
    base = to_rgb(gray)
    m = cv2.resize(mask128, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = base.copy()
    for c, col in COLORS.items():
        if c in show:
            sel = m == c
            out[sel] = (1 - alpha) * base[sel] + alpha * (np.array(col) / 255.0)
    return out


def region_dice(gt, pr):
    out = {}
    for n, fn in REGIONS.items():
        a, b = fn(gt), fn(pr)
        out[n] = 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + 1e-6)
    return out


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧠 NeuroScan U-Net")
model_path = st.sidebar.text_input("Model file", "unet_brats.keras")
model_up = st.sidebar.file_uploader("...or upload model (.keras / .h5)", type=["keras", "h5"])
if model_up is not None:
    ext = os.path.splitext(model_up.name)[1]
    tmp = os.path.join(tempfile.gettempdir(), "uploaded_model" + ext)
    with open(tmp, "wb") as fh:
        fh.write(model_up.getvalue())
    model_path = tmp
page = st.sidebar.radio("Navigate", ["Upload & Predict", "Segmentation Viewer",
                                     "Analytics", "3D View", "Model Performance", "About"])
st.sidebar.warning("Educational project only. Not a medical device. "
                   "Never use for diagnosis or treatment decisions.")

S = st.session_state


# ------------------------------------------------------------------ pages
def page_upload():
    st.header("Upload MRI scans")
    st.caption("Upload 1-4 scans (NIfTI .nii/.nii.gz, or PNG/JPG single slices). "
               "The model works with any subset, but accuracy is best with all four.")
    files = st.file_uploader("Scans", type=["nii", "gz", "png", "jpg", "jpeg"],
                             accept_multiple_files=True)
    if not files:
        return
    if len(files) > 4:
        st.error("Please upload at most 4 scans.")
        return

    st.subheader("Scan types")
    assign, cols = {}, st.columns(len(files))
    for col, f in zip(cols, files):
        guess = detect_modality(f.name)
        idx = MODALITIES.index(guess) if guess else 0
        with col:
            assign[f.name] = st.selectbox(f.name, MODALITIES, index=idx, key="mod_" + f.name)
            if guess is None:
                st.caption("⚠️ couldn't detect from filename - please check")
    if len(set(assign.values())) != len(files):
        st.error("Each scan must have a different type.")
        return

    gt_file = st.file_uploader("Optional: ground-truth segmentation (BraTS *_seg.nii) "
                               "to compute Dice", type=["nii", "gz"])

    if st.button("Run prediction", type="primary"):
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return
        vols, spacing = {}, None
        for f in files:
            v, z = read_upload(f)
            vols[assign[f.name]] = v
            spacing = z if spacing is None else spacing
        shapes = {v.shape for v in vols.values()}
        if len(shapes) != 1:
            st.error(f"Scans have different shapes: {shapes}. They must be co-registered.")
            return
        with st.spinner("Segmenting every slice..."):
            pred, conf, secs = predict_volume(model, vols)
        S.update(vols=vols, spacing=spacing, pred=pred, conf=conf, secs=secs, gt=None)
        if gt_file is not None:
            g, _ = read_upload(gt_file)
            g = np.rint(g).astype(np.uint8)
            g[g == 4] = 3
            gt128 = np.stack([cv2.resize(np.rot90(g[:, :, z]).copy(), (IMG, IMG),
                                         interpolation=cv2.INTER_NEAREST)
                              for z in range(g.shape[2])])
            S["gt"] = gt128
        st.success(f"Done in {secs:.1f}s. Open the **Segmentation Viewer** from the sidebar.")

    if "pred" in S:
        st.info(f"A result is loaded ({len(S['vols'])} scan(s): {', '.join(S['vols'])}).")
        if len(S["vols"]) < 4:
            st.warning("Missing scan types - segmentation of tumour sub-regions "
                       "(especially T1-only or T2-only) is much less reliable.")


def need_result():
    if "pred" not in S:
        st.info("Upload scans and run a prediction first.")
        return False
    return True


def page_viewer():
    st.header("Segmentation viewer")
    if not need_result():
        return
    vols, pred = S["vols"], S["pred"]
    depth = pred.shape[0]
    areas = (pred > 0).reshape(depth, -1).sum(1)
    default = int(areas.argmax())
    z = st.slider("Axial slice", 0, depth - 1, default) if depth > 1 else 0
    c1, c2 = st.columns(2)
    alpha = c1.slider("Mask opacity", 0.0, 1.0, 0.6, 0.05)
    show = c2.multiselect("Classes shown", [1, 2, 3], default=[1, 2, 3],
                          format_func=lambda c: CLASS_NAMES[c])

    st.subheader("All scans")
    cols = st.columns(len(vols))
    for col, (m, v) in zip(cols, vols.items()):
        col.image(to_rgb(rot(v, z)), caption=m, use_container_width=True)

    st.subheader("Prediction")
    ref = "FLAIR" if "FLAIR" in vols else list(vols)[0]
    ref = st.selectbox("Background scan", list(vols), index=list(vols).index(ref))
    base = rot(vols[ref], z)
    a, b, c = st.columns(3)
    a.image(to_rgb(base), caption=f"{ref} (original)", use_container_width=True)
    a_ov = overlay(base, pred[z], alpha, show)
    b.image(a_ov, caption="Overlay", use_container_width=True)
    mask_rgb = np.zeros((IMG, IMG, 3))
    for k, col in COLORS.items():
        mask_rgb[pred[z] == k] = np.array(col) / 255.0
    c.image(mask_rgb, caption="Predicted mask", use_container_width=True, clamp=True)
    st.markdown("🟥 Necrotic core &nbsp; 🟩 Oedema &nbsp; 🟨 Enhancing tumour")

    if S.get("gt") is not None:
        st.subheader("Ground truth vs prediction")
        g1, g2 = st.columns(2)
        g1.image(overlay(base, S["gt"][z], alpha, show), caption="Ground truth",
                 use_container_width=True)
        g2.image(a_ov, caption="Prediction", use_container_width=True)

    tumour = pred[z] > 0
    if tumour.any():
        st.caption(f"Mean model confidence on tumour pixels (this slice): "
                   f"{S['conf'][z][tumour].mean():.2%}")
    else:
        st.caption("No tumour predicted on this slice.")


def page_analytics():
    st.header("Tumour analytics")
    if not need_result():
        return
    pred, sp = S["pred"], S["spacing"]
    depth = pred.shape[0]
    h, w = next(iter(S["vols"].values())).shape[:2]
    # counts happen on the 128x128 grid; rescale to original voxel count
    scale = (h * w) / (IMG * IMG)
    vox_ml = float(np.prod(sp)) / 1000.0

    rows = []
    for c in (1, 2, 3):
        n = int((pred == c).sum() * scale)
        rows.append({"Class": CLASS_NAMES[c], "Voxels": n,
                     "Volume (mL)": round(n * vox_ml, 2) if depth > 1 else None})
    df = pd.DataFrame(rows)
    tot = df["Voxels"].sum()
    df["% of tumour"] = (100 * df["Voxels"] / max(tot, 1)).round(1)
    st.dataframe(df, hide_index=True, use_container_width=True)
    if depth == 1:
        st.caption("Single 2D image uploaded - volume isn't available, showing pixel counts.")
    else:
        st.caption(f"Voxel spacing from file header: {sp.round(2)} mm. "
                   "Volumes are model estimates, not clinical measurements.")

    reg = {n: int(fn(pred).sum() * scale) for n, fn in REGIONS.items()}
    reg_df = pd.DataFrame({"Region": reg.keys(), "Voxels": reg.values()})
    st.bar_chart(reg_df, x="Region", y="Voxels")

    if depth > 1:
        st.subheader("Tumour area per slice")
        st.line_chart(pd.DataFrame({"tumour pixels": (pred > 0).reshape(depth, -1).sum(1)}))

    if S.get("gt") is not None:
        st.subheader("Dice vs ground truth")
        d = region_dice(S["gt"], pred)
        st.dataframe(pd.DataFrame({"Region": d.keys(), "Dice": np.round(list(d.values()), 3)}),
                     hide_index=True)

    st.metric("Inference time", f"{S['secs']:.1f} s for {depth} slice(s)")
    report = (f"Brain tumour segmentation report (educational)\n"
              f"Scans: {', '.join(S['vols'])}\n\n{df.to_string(index=False)}\n\n"
              f"Regions (voxels): {reg}\n\nNot a medical device.")
    st.download_button("Download report (.txt)", report, "report.txt")
    st.download_button("Download table (.csv)", df.to_csv(index=False), "tumour_stats.csv")


def page_3d():
    st.header("3D tumour view")
    if not need_result():
        return
    pred = S["pred"]
    if pred.shape[0] < 5:
        st.info("3D view needs a full volume (NIfTI), not a single slice.")
        return
    fig = go.Figure()
    rng = np.random.default_rng(0)
    names = {1: "Necrotic core", 2: "Oedema", 3: "Enhancing tumour"}
    cols = {1: "red", 2: "green", 3: "gold"}
    for c in (1, 2, 3):
        idx = np.argwhere(pred == c)
        if len(idx) == 0:
            continue
        if len(idx) > 6000:
            idx = idx[rng.choice(len(idx), 6000, replace=False)]
        fig.add_trace(go.Scatter3d(x=idx[:, 2], y=idx[:, 1], z=idx[:, 0], mode="markers",
                                   marker=dict(size=2, color=cols[c], opacity=0.5 if c == 2 else 0.9),
                                   name=names[c]))
    fig.update_layout(height=650, scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Point cloud of predicted voxels (subsampled for speed).")


def page_perf():
    st.header("Model performance (from training notebook)")
    st.caption("Test Dice on 12 unseen BraTS 2020 patients.")
    df = pd.DataFrame(
        {"WT (whole tumour)": [0.908, 0.656, 0.732, 0.801, 0.877],
         "TC (tumour core)": [0.865, 0.356, 0.755, 0.416, 0.653],
         "ET (enhancing tumour)": [0.807, 0.007, 0.754, 0.166, 0.413]},
        index=["All 4 scans", "T1 only", "T1ce only", "T2 only", "FLAIR only"])
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df)
    st.info("Takeaway: use all four scans if you can. If you only have one, T1ce or FLAIR "
            "work best; T1 alone is poor for enhancing tumour.")


def page_about():
    st.header("About")
    st.markdown("""
- **Model:** U-Net, input 128×128×4 (T1, T1ce, T2, FLAIR), trained on BraTS 2020 with *modality dropout*.
- **Output:** per-pixel class: background, necrotic core, oedema, enhancing tumour.
- **Note:** this model **segments** tumours; it does not classify tumour *type* (e.g. glioma vs meningioma).
- Scans must be co-registered and skull-stripped like BraTS data for good results.
- ⚠️ Educational only; not for diagnosis.
""")


{"Upload & Predict": page_upload, "Segmentation Viewer": page_viewer,
 "Analytics": page_analytics, "3D View": page_3d,
 "Model Performance": page_perf, "About": page_about}[page]()
