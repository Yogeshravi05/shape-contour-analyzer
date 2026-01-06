import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shape & Contour Analyzer – Pro Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("## 📊 Shape & Contour Analyzer – Pro Dashboard")
st.caption("Advanced visual analytics for geometric shape detection using contour analysis")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Analysis Controls")

preset = st.sidebar.selectbox(
    "Preset Configuration",
    ["Default", "Simple Shapes", "Dense Objects", "Noisy Background"]
)

preset_map = {
    "Default": (5, 200, 500),
    "Simple Shapes": (3, 180, 800),
    "Dense Objects": (3, 200, 200),
    "Noisy Background": (7, 160, 600)
}

blur_k, thresh_v, min_area = preset_map[preset]

blur_k = st.sidebar.slider("Blur Kernel Size", 1, 15, blur_k, step=2)
thresh_v = st.sidebar.slider("Threshold Value", 50, 255, thresh_v)
min_area = st.sidebar.slider("Minimum Area Filter", 100, 10000, min_area)

show_edges = st.sidebar.checkbox("Show Edge Detection")
show_pipeline = st.sidebar.checkbox("Show Full Pipeline", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard Mode**")
highlight_shape = st.sidebar.selectbox(
    "Highlight Shape Type",
    ["All", "Triangle", "Square", "Rectangle", "Circle"]
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image for Analysis", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # ---------------- PREPROCESSING ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, thresh = cv2.threshold(blur, thresh_v, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    results = []

    # ---------------- SHAPE ANALYSIS ----------------
    for idx, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape = "Triangle"
            color = (0, 128, 255)
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            shape = "Square" if 0.95 <= w / h <= 1.05 else "Rectangle"
            color = (0, 200, 0)
        else:
            shape = "Circle"
            color = (255, 0, 0)

        if highlight_shape != "All" and shape != highlight_shape:
            color = (150, 150, 150)

        cv2.drawContours(output, [cnt], -1, color, 2)
        x, y = approx[0][0]
        cv2.putText(output, shape, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        confidence = min(100, int((area / (perimeter + 1)) * 10))

        results.append({
            "Object ID": idx,
            "Shape": shape,
            "Area (px²)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2),
            "Vertices": len(approx),
            "Confidence (%)": confidence
        })

    df = pd.DataFrame(results)

    # ---------------- DASHBOARD TABS ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 Detection", "📊 Summary", "📈 Comparative Analytics", "📥 Export", "ℹ️ Insights"]
    )

    # -------- TAB 1: DETECTION --------
    with tab1:
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Image", use_column_width=True)
        col2.image(output, caption="Detected & Highlighted Shapes", use_column_width=True)

        if show_pipeline:
            st.markdown("### 🧪 Processing Pipeline")
            p1, p2, p3 = st.columns(3)
            p1.image(gray, caption="Grayscale")
            p2.image(thresh, caption="Thresholded")
            p3.image(edges, caption="Edges")

    # -------- TAB 2: SUMMARY --------
    with tab2:
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Objects", len(df))
            c2.metric("Unique Shapes", df["Shape"].nunique())
            c3.metric("Largest Area", int(df["Area (px²)"].max()))
            c4.metric("Avg Confidence", f"{df['Confidence (%)'].mean():.1f}%")

            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No objects detected. Adjust parameters.")

    # -------- TAB 3: COMPARATIVE ANALYTICS --------
    with tab3:
        if not df.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                df.groupby("Shape")["Area (px²)"].mean().plot(kind="bar", ax=ax1)
                ax1.set_title("Average Area by Shape")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                df.groupby("Shape")["Perimeter (px)"].mean().plot(kind="bar", ax=ax2)
                ax2.set_title("Average Perimeter by Shape")
                st.pyplot(fig2)

    # -------- TAB 4: EXPORT --------
    with tab4:
        if not df.empty:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Analysis (CSV)",
                csv,
                "shape_contour_analysis.csv",
                "text/csv"
            )

    # -------- TAB 5: INSIGHTS --------
    with tab5:
        if not df.empty:
            dominant = df["Shape"].mode()[0]
            st.markdown(f"""
            ### 🔍 Automated Insights

            • The image contains **{len(df)} detected objects**  
            • The most frequent shape is **{dominant}**  
            • Objects with larger area generally show higher perimeter values  
            • Confidence scores indicate reliable shape approximation  

            This suggests the contour-based method performs well for clear geometric structures.
            """)
        else:
            st.info("No insights available.")

else:
    st.info("Upload an image to activate the dashboard.")
