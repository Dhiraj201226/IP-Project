import streamlit as st
import cv2
import os
import sys
import numpy as np
from PIL import Image

# Ensure current working directory and sys.path are set to this file's folder (IP_project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
try:
    os.chdir(BASE_DIR)
except Exception:
    pass

from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline
from pipeline.similarity_pipeline import SimilarityPipeline

st.set_page_config(page_title="NOCTIVISION", layout="wide", page_icon="👁️")

# Custom CSS to bound images and prevent screen blowup
st.markdown("""
<style>
    .stImage img {
        max-width: 600px !important;
        max-height: 600px !important;
        object-fit: contain;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def prepare_and_resize(image, target_size=(600, 600)):
    """
    Standardizes image to exactly 600x600 px before computation and display.
    Minimizes computational load and ensures uniform display and bounding box coordinates.
    """
    if image is None:
        return None
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

st.title("NOCTIVISION: Low-Light Vision & Person Matching System")
st.markdown("Low-Light Image Enhancement, Crowd Density Estimation, Person Detection & Cosine Similarity Ranking")

# Sidebar Navigation Mode
app_mode = st.sidebar.selectbox(
    "Select System Mode",
    ["Surveillance Pipeline", "Person & Face Similarity Ranking"]
)

# Resolution Standardization in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("⚡ Resolution Standardization")
st.sidebar.info("📐 All images are automatically resized to **600×600 px** before processing to save computation and prevent screen blowup.")
st.sidebar.markdown("---")

@st.cache_resource
def get_surveillance_pipelines():
    return ImagePipeline(), VideoPipeline()

@st.cache_resource
def get_similarity_pipeline(model_name):
    return SimilarityPipeline(model_name=model_name)

if app_mode == "Surveillance Pipeline":
    pipeline, video_pipeline = get_surveillance_pipelines()

    st.sidebar.header("Pipeline Options")
    enable_enh = st.sidebar.checkbox("Low-Light Enhancement", value=True)
    enable_sr = st.sidebar.checkbox(
        "Super Resolution",
        value=False,
        help="Disabled by default. When enabled, uses fast 10ms edge-preserving sharpening."
    )
    sr_method = "fast"
    if enable_sr:
        sr_choice = st.sidebar.selectbox(
            "Super Resolution Engine",
            ["Fast Edge-Preserving Sharpening (~0.02s)", "Deep EDSR Neural Network (Slow, ~4 mins on CPU)"],
            index=0
        )
        sr_method = "edsr" if "EDSR" in sr_choice else "fast"

    enable_det = st.sidebar.checkbox("Person Detection", value=True)
    enable_crowd = st.sidebar.checkbox("Crowd Analysis", value=True)

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Detection Sensitivity")
    conf_thresh = st.sidebar.slider(
        "Person Confidence Threshold",
        min_value=0.10,
        max_value=0.70,
        value=0.20,
        step=0.05,
        help="Lower threshold (0.15–0.25) detects crowded, occluded, and dark background people. Default is 0.20 for nighttime crowds."
    )

    pipeline.config['enhancement']['enabled'] = enable_enh
    pipeline.config['super_resolution']['enabled'] = enable_sr
    pipeline.config['super_resolution']['method'] = sr_method
    pipeline.config['detection']['enabled'] = enable_det
    pipeline.config['crowd_counting']['enabled'] = enable_crowd
    pipeline.config['runtime']['confidence_threshold'] = conf_thresh

    video_pipeline.config['enhancement']['enabled'] = enable_enh
    video_pipeline.config['super_resolution']['enabled'] = enable_sr
    video_pipeline.config['super_resolution']['method'] = sr_method
    video_pipeline.config['tracking']['enabled'] = enable_det
    video_pipeline.config['crowd_counting']['enabled'] = enable_crowd
    video_pipeline.config['runtime']['confidence_threshold'] = conf_thresh

    input_type = st.radio("Select Input Type", ["Image", "Video"])

    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            temp_path = os.path.join("data", "test", "images", "temp_upload.jpg")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            # Decode and resize BEFORE saving and displaying (so less computation!)
            file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
            raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            resized_img = prepare_and_resize(raw_img)
            cv2.imwrite(temp_path, resized_img)

            st.image(
                cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB),
                caption="Input Image (Standardized: 600x600 px)",
                width=600
            )

            if st.button("Run Pipeline"):
                with st.spinner("Processing..."):
                    results = pipeline.run(temp_path)

                    st.success(f"Processing Complete in {results['processing_time']:.2f} seconds!")

                    col1, col2 = st.columns(2)
                    if 'enhanced' in results:
                        enh_rgb = cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB)
                        col1.image(enh_rgb, caption=f"Enhanced Image ({enh_rgb.shape[1]}x{enh_rgb.shape[0]} px)", width=min(500, enh_rgb.shape[1]))
                    if 'super_resolved' in results:
                        sr_rgb = cv2.cvtColor(results['super_resolved'], cv2.COLOR_BGR2RGB)
                        col2.image(sr_rgb, caption=f"Super Resolved Image ({sr_rgb.shape[1]}x{sr_rgb.shape[0]} px)", width=min(500, sr_rgb.shape[1]))

                    final_rgb = cv2.cvtColor(results['final_image'], cv2.COLOR_BGR2RGB)
                    st.image(final_rgb, caption=f"Final Detection Result ({final_rgb.shape[1]}x{final_rgb.shape[0]} px)", width=min(600, final_rgb.shape[1]))

                    st.subheader("Analysis")
                    st.write(f"**People Detected:** {results.get('num_detected', 'N/A')}")
                    st.write(f"**Estimated Crowd:** {results.get('crowd_count', 'N/A')}")
                    st.write(f"**Density:** {results.get('crowd_density', 'N/A')}")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])

        if uploaded_file is not None:
            temp_video_path = os.path.join("data", "test", "videos", "temp_upload.mp4")
            output_video_path = os.path.join("data", "test", "videos", "output_video.webm")
            os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.video(temp_video_path)

            if st.button("Run Video Pipeline"):
                with st.spinner("Processing Video (this may take a while)..."):
                    results = video_pipeline.run(temp_video_path, output_video_path)

                    st.success(f"Processing Complete in {results['total_processing_time']:.2f} seconds!")
                    st.video(output_video_path)

                    st.subheader("Analysis")
                    st.write(f"**Average FPS:** {results.get('average_fps', 0):.2f}")
                    st.write(f"**Frames Processed:** {results.get('frames_processed', 0)}")
                    st.write(f"**Unique Tracks (People):** {results.get('unique_tracks', 'N/A')}")

elif app_mode == "Person & Face Similarity Ranking":
    st.subheader("🎯 Reference-Based Person / Face Matching & Ranking")
    st.markdown(
        "Upload the individual person you want to find and the group/scene photo. "
        "Select the search direction to find the person inside the scene."
    )

    st.sidebar.header("Matcher Settings")
    feature_model_choice = st.sidebar.selectbox(
        "Person Feature Model",
        [
            "Deep Part-Based Re-ID CNN (Recommended - Multi-Part + Color)",
            "ResNet-18 Deep CNN (Global Features)",
            "YOLOv8 Nano Backbone"
        ],
        index=0,
        help="Deep Part-Based Re-ID decomposes the body into head, shirt/torso, and pants, extracting deep CNN features and chromatic profiles to match people accurately even in crowds."
    )
    if "Part-Based" in feature_model_choice:
        selected_model = "part_reid"
    elif "ResNet" in feature_model_choice:
        selected_model = "resnet18"
    else:
        selected_model = "yolov8n.pt"

    match_mode = st.sidebar.radio(
        "Detection Entity",
        ["Face (Isolate & Check Face Region Only)", "Person (Full Body)"],
        index=0,
        help="Selecting 'Face' cleans the image and isolates face regions specifically, avoiding confusion from clothing or background."
    )
    enable_enhance = st.sidebar.checkbox(
        "Clean & Enhance Low-Light Images",
        value=True,
        help="Applies CLAHE contrast enhancement and edge-preserving bilateral denoising to bring out facial contours in the dark."
    )

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader(
            "📷 Image 1",
            type=['jpg', 'jpeg', 'png'],
            key="file1_uploader",
            help="Upload the individual person or the group/scene photo."
        )
    with col2:
        file2 = st.file_uploader(
            "📷 Image 2",
            type=['jpg', 'jpeg', 'png'],
            key="file2_uploader",
            help="Upload the other image (individual person or group/scene photo)."
        )

    img1_cv2 = None
    img2_cv2 = None

    if file1:
        b1 = np.frombuffer(file1.getvalue(), dtype=np.uint8)
        raw1 = cv2.imdecode(b1, cv2.IMREAD_COLOR)
        if raw1 is not None:
            img1_cv2 = prepare_and_resize(raw1)
            with col1:
                st.image(
                    cv2.cvtColor(img1_cv2, cv2.COLOR_BGR2RGB),
                    caption="Image 1 (Standardized: 600x600 px)",
                    width=450
                )

    if file2:
        b2 = np.frombuffer(file2.getvalue(), dtype=np.uint8)
        raw2 = cv2.imdecode(b2, cv2.IMREAD_COLOR)
        if raw2 is not None:
            img2_cv2 = prepare_and_resize(raw2)
            with col2:
                st.image(
                    cv2.cvtColor(img2_cv2, cv2.COLOR_BGR2RGB),
                    caption="Image 2 (Standardized: 600x600 px)",
                    width=450
                )

    if img1_cv2 is not None and img2_cv2 is not None:
        st.markdown("---")
        st.write("### 🧭 Search Direction Configuration")
        direction = st.radio(
            "Which image contains the person you want to find?",
            [
                "👤 Target Person is in Image 1 ➔ Search inside Scene Image 2",
                "👤 Target Person is in Image 2 ➔ Search inside Scene Image 1"
            ],
            index=0
        )

        if "Image 1 ➔ Search inside Scene Image 2" in direction:
            query_img = img1_cv2
            scene_img = img2_cv2
            query_name = "Image 1 (Person)"
            scene_name = "Image 2 (Scene)"
        else:
            query_img = img2_cv2
            scene_img = img1_cv2
            query_name = "Image 2 (Person)"
            scene_name = "Image 1 (Scene)"

        entity_mode = "face" if "Face" in match_mode else "person"
        st.info(f"Searching for **{entity_mode}** of person in **{query_name}** inside **{scene_name}** using **{feature_model_choice}**.")

        if st.button("🚀 Find & Rank Matching Persons", type="primary"):
            with st.spinner(f"Cleaning images and matching {entity_mode} regions in {scene_name}..."):
                sim_pipeline = get_similarity_pipeline(selected_model)
                results = sim_pipeline.run(
                    query_img,
                    scene_img,
                    mode=entity_mode,
                    enhance=enable_enhance
                )

                st.success(
                    f"Analyzed {results['total_detected']} candidate(s) in {scene_name} in {results['processing_time']:.2f} seconds!"
                )

                top_match = results.get("top_match")
                query_crop = results.get("query_crop")

                if top_match is not None:
                    # Comparison view: Query Person vs Matched Candidate
                    st.markdown("---")
                    st.subheader("🏆 Match Result: Target Person vs. Best Found Candidate")

                    cmp_col1, cmp_col2, cmp_col3 = st.columns([1, 1, 1.5])
                    with cmp_col1:
                        if query_crop is not None:
                            q_rgb = cv2.cvtColor(query_crop, cv2.COLOR_BGR2RGB)
                            st.image(
                                q_rgb,
                                caption=f"Target Person ({query_name})",
                                width=min(250, max(160, q_rgb.shape[1]))
                            )
                    with cmp_col2:
                        top_crop_rgb = cv2.cvtColor(top_match["crop"], cv2.COLOR_BGR2RGB)
                        ch, cw = top_crop_rgb.shape[:2]
                        st.image(
                            top_crop_rgb,
                            caption=f"Best Match in {scene_name} (Rank #1)",
                            width=min(250, max(160, cw))
                        )
                    with cmp_col3:
                        st.metric(
                            label="Cosine Similarity Score",
                            value=f"{top_match['percentage']:.2f}%",
                            delta=f"Raw cosine: {top_match['score']:.4f}"
                        )
                        st.info(f"**Bounding Box:** `[x1={top_match['bbox'][0]}, y1={top_match['bbox'][1]}, x2={top_match['bbox'][2]}, y2={top_match['bbox'][3]}]`")
                        st.markdown(
                            f"The highlighted candidate in **{scene_name}** has the highest feature similarity with the target person."
                        )

                    st.markdown("---")
                    st.subheader(f"🖼️ {scene_name} with Ranked Detections")
                    if results.get("annotated_image") is not None:
                        annotated_rgb = cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB)
                        ah, aw = annotated_rgb.shape[:2]
                        st.image(
                            annotated_rgb,
                            caption=f"{scene_name} Annotated (Green Box = #1 Match) - {aw}x{ah} px",
                            width=min(700, aw)
                        )

                    # Display Ranked Gallery of all candidates
                    st.markdown("---")
                    st.subheader("📊 Ranked Candidate Gallery (Highest to Lowest Similarity)")
                    candidates = results["ranked_candidates"]

                    cols_per_row = 4
                    for row_start in range(0, len(candidates), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for col_idx, cand in enumerate(candidates[row_start:row_start + cols_per_row]):
                            with cols[col_idx]:
                                crop_rgb = cv2.cvtColor(cand["crop"], cv2.COLOR_BGR2RGB)
                                st.image(crop_rgb, width=150)
                                is_top = cand["rank"] == 1
                                title = f"🥇 **Rank #{cand['rank']}**" if is_top else f"**Rank #{cand['rank']}**"
                                st.markdown(title)
                                st.write(f"Similarity: **{cand['percentage']:.1f}%**")
                                st.caption(f"Raw Cosine: `{cand['score']:.4f}`")
                else:
                    st.warning(f"No candidates detected in {scene_name}. Try adjusting lighting or toggling detection entity (Person/Face) in sidebar.")
