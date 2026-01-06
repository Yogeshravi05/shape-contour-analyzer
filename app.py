import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="🔍",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.metric-label {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Adjust parameters to improve detection")

blur_kernel = st.sidebar.slider("Gaussian Blur Kernel", 1, 15, 5, step=2)
threshold_val = st.sidebar.slider("Threshold Value", 50, 255, 200)
min_area = st.sidebar.slider("Minimum Contour Area", 100, 10000, 500)
epsilon_factor = st.sidebar.slider("Shape Approximation Accuracy", 1, 10, 4)

show_edges = st.sidebar.checkbox("Show Edge Detection")
show_contours = st.sidebar.checkbox("Show Contours", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Built using OpenCV + Streamlit")

# -------------------- TITLE --------------------
st.title("🔍 Shape & Contour Analyzer")
st.markdown(
    "An **interactive computer vision dashboard** to detect shapes, count objects, "
    "and analyze geometric features such as **area and perimeter**."
)

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # -------------------- PREPROCESSING --------------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, thresh = cv2.threshold(
        blur, threshold_val, 255, cv2.THRESH_BINARY_INV
    )

    edges = cv2.Canny(blur, 100, 200)

    # -------------------- CONTOUR DETECTION --------------------
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    results = []

    # -------------------- SHAPE ANALYSIS --------------------
    for idx, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = (epsilon_factor / 100) * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Shape Classification
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Unknown"

        # Draw Contours
        if show_contours:
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            x, y = approx[0][0]
            cv2.putText(
                output, shape,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2
            )

        results.append({
            "Object ID": idx,
            "Shape": shape,
            "Area (px²)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2),
            "Vertices": len(approx)
        })

    # -------------------- METRICS --------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("🧮 Total Objects", len(results))
    col2.metric("📐 Avg Area", round(np.mean([r["Area (px²)"] for r in results]), 2) if results else 0)
    col3.metric("📏 Avg Perimeter", round(np.mean([r["Perimeter (px)"] for r in results]), 2) if results else 0)

    # -------------------- IMAGE DISPLAY --------------------
    st.markdown("### 🖼️ Processed Output")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.image(img, caption="Original Image", use_column_width=True)

    with img_col2:
        st.image(output, caption="Detected Shapes", use_column_width=True)

    if show_edges:
        st.markdown("### ✴️ Edge Detection")
        st.image(edges, caption="Canny Edges", use_column_width=True)

    # -------------------- DATA TABLE --------------------
    if results:
        df = pd.DataFrame(results)
        st.markdown("### 📊 Shape Analysis Table")
        st.dataframe(df, use_container_width=True)

        # -------------------- CHARTS --------------------
        st.markdown("### 📈 Visual Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1, ax1 = plt.subplots()
            df["Shape"].value_counts().plot(kind="bar", ax=ax1)
            ax1.set_title("Shape Distribution")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        with chart_col2:
            fig2, ax2 = plt.subplots()
            ax2.hist(df["Area (px²)"], bins=10)
            ax2.set_title("Area Distribution")
            ax2.set_xlabel("Area")
            st.pyplot(fig2)

        # -------------------- EXPORT --------------------
        st.markdown("### 📥 Export Results")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Report",
            data=csv,
            file_name="shape_analysis.csv",
            mime="text/csv"
        )

    else:
        st.warning("No valid shapes detected. Try adjusting parameters.")

else:
    st.info("Please upload an image to start analysis.")
