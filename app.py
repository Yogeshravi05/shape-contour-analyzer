import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🔍 Shape & Contour Analyzer")
st.caption("An interactive computer vision dashboard for shape detection and feature extraction")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

preset = st.sidebar.selectbox(
    "Detection Preset",
    ["Default", "Simple Shapes", "Small Objects", "Noisy Image"]
)

if preset == "Simple Shapes":
    blur_k, thresh_v, min_area = 3, 180, 800
elif preset == "Small Objects":
    blur_k, thresh_v, min_area = 3, 200, 200
elif preset == "Noisy Image":
    blur_k, thresh_v, min_area = 7, 160, 600
else:
    blur_k, thresh_v, min_area = 5, 200, 500

blur_k = st.sidebar.slider("Gaussian Blur Kernel", 1, 15, blur_k, step=2)
thresh_v = st.sidebar.slider("Threshold Value", 50, 255, thresh_v)
min_area = st.sidebar.slider("Minimum Object Area", 100, 10000, min_area)

show_edges = st.sidebar.checkbox("Show Edge Detection")
show_pipeline = st.sidebar.checkbox("Show Processing Pipeline", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Developed as part of Computer Vision coursework")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # ---------------- PROCESSING ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, thresh = cv2.threshold(blur, thresh_v, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    results = []

    for i, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape = "Triangle"
            color = (255, 0, 0)
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            shape = "Square" if 0.95 <= w/h <= 1.05 else "Rectangle"
            color = (0, 255, 0)
        else:
            shape = "Circle"
            color = (0, 0, 255)

        cv2.drawContours(output, [cnt], -1, color, 2)
        x, y = approx[0][0]
        cv2.putText(output, shape, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results.append({
            "Object ID": i,
            "Shape": shape,
            "Area (px²)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2),
            "Vertices": len(approx)
        })

    df = pd.DataFrame(results)

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Detection", "📊 Analytics", "📥 Export", "ℹ️ About"]
    )

    # -------- TAB 1: DETECTION --------
    with tab1:
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Image", use_column_width=True)
        col2.image(output, caption="Detected Shapes", use_column_width=True)

        if show_pipeline:
            st.markdown("### 🧪 Processing Pipeline")
            p1, p2, p3 = st.columns(3)
            p1.image(gray, caption="Grayscale")
            p2.image(thresh, caption="Thresholded")
            p3.image(edges, caption="Edges")

    # -------- TAB 2: ANALYTICS --------
    with tab2:
        if not df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Objects", len(df))
            c2.metric("Most Common Shape", df["Shape"].mode()[0])
            c3.metric("Largest Area", int(df["Area (px²)"].max()))

            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots()
            df["Shape"].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Shape Distribution")
            st.pyplot(fig)
        else:
            st.warning("No objects detected. Adjust parameters.")

    # -------- TAB 3: EXPORT --------
    with tab3:
        if not df.empty:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Analysis CSV",
                csv,
                "shape_analysis.csv",
                "text/csv"
            )

    # -------- TAB 4: ABOUT --------
    with tab4:
        st.markdown("""
        **Shape & Contour Analyzer**

        - Detects geometric shapes using contour analysis  
        - Extracts area and perimeter features  
        - Visualizes full image-processing pipeline  
        - Deployed using Streamlit Cloud  

        **Developed by:** Yogesh Ravi M  
        **Technology:** Python | OpenCV | Streamlit
        """)

else:
    st.info("Upload an image to begin analysis.")

