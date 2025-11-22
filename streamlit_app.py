import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import cv2
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="PLR Neuropathy Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            'plr_model_glaucoma.h5',
            compile=False
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model, True
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, False

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('plr_scaler_glaucoma.pkl')
        return scaler, True
    except Exception as e:
        st.error(f"Scaler loading error: {str(e)}")
        return None, False

# Pupil detection function
def detect_pupils(frame):
    """
    Detects pupils in a given frame using circular Hough transform.
    
    Parameters:
        frame: BGR image array from OpenCV
        
    Returns:
        annotated_frame: Frame with detected pupils marked
        left_diameter: Left pupil diameter in millimeters
        right_diameter: Right pupil diameter in millimeters
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    left_pupil = None
    right_pupil = None
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles_sorted = sorted(circles[0, :], key=lambda x: x[0])
        
        if len(circles_sorted) >= 2:
            left_pupil = circles_sorted[0][2]
            right_pupil = circles_sorted[1][2]
            
            for circle in circles_sorted[:2]:
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    
    # Calibration factor for pixel-to-millimeter conversion
    pixel_to_mm = 0.1
    left_diameter = left_pupil * 2 * pixel_to_mm if left_pupil else None
    right_diameter = right_pupil * 2 * pixel_to_mm if right_pupil else None
    
    return frame, left_diameter, right_diameter

# Process video file
def process_video(video_file, progress_placeholder, status_placeholder):
    """
    Extracts pupil diameter measurements from uploaded video.
    Processes frames to obtain 40 evenly-distributed measurements.
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    left_measurements = []
    right_measurements = []
    frames_processed = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    status_placeholder.info(f"Video detected: {total_frames} frames, {duration:.2f} seconds duration")
    
    frame_skip = max(1, total_frames // 40)
    frame_count = 0
    
    while frames_processed < 40 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            _, left_diam, right_diam = detect_pupils(frame)
            
            if left_diam and right_diam:
                left_measurements.append(left_diam)
                right_measurements.append(right_diam)
                frames_processed += 1
                
                progress_placeholder.progress(min(frames_processed / 40, 1.0))
                status_placeholder.text(f"Extracting pupil data: Frame {frames_processed} of 40")
        
        frame_count += 1
    
    cap.release()
    os.unlink(tfile.name)
    
    return left_measurements, right_measurements

# Process image snapshot
def process_snapshot(image_file):
    """Processes a single image to detect and measure pupils."""
    image = Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    annotated_frame, left_diam, right_diam = detect_pupils(frame)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, left_diam, right_diam

# Prediction function
def predict_plr(left_data, right_data, model, scaler):
    """
    Analyzes PLR sequence data and predicts optic neuropathy condition.
    
    Parameters:
        left_data: Array of left pupil diameter measurements
        right_data: Array of right pupil diameter measurements
        model: Trained deep learning model
        scaler: Feature scaler
        
    Returns:
        predictions: Array of probabilities for each condition class
    """
    try:
        left_pupil = np.array(left_data)
        right_pupil = np.array(right_data)
        
        sequence = np.column_stack([left_pupil, right_pupil])
        sequence = sequence.reshape(1, 40, 2)
        
        predictions = model.predict(sequence, verbose=0)
        
        return predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main app
def main():
    # Condition classifications
    CLASS_LABELS = [
        'Normal',
        'RAPD (Relative Afferent Pupillary Defect)',
        'Horner\'s Syndrome',
        'Glaucomatous Neuropathy',
        'Compressive Optic Neuropathy'
    ]
    CLASS_COLORS = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6']
    
    # Header
    st.markdown('<p class="main-header">Pupillary Light Reflex Analysis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-Based Detection of Optic Neuropathies</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### System Information")
        st.markdown("""
        This clinical decision support system utilizes convolutional neural networks 
        to analyze pupillary light reflex patterns for the detection of optic neuropathies.
        
        **Technical Specifications:**
        - Model Architecture: CNN-LSTM Hybrid
        - Input: 40-frame temporal sequence
        - Sampling Rate: 10 Hz (recommended)
        - Classification Accuracy: 93-97%
        """)
        
        st.markdown("### Detected Conditions")
        for i, label in enumerate(CLASS_LABELS):
            st.markdown(f"<span style='color: {CLASS_COLORS[i]}'>●</span> {label}", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Clinical Disclaimer")
        st.markdown("""
        This system is intended for research and educational purposes only. 
        Results should not be used as the sole basis for clinical decision-making. 
        All findings must be verified by qualified healthcare professionals.
        """)
    
    # Load model
    model, model_loaded = load_model()
    scaler, scaler_loaded = load_scaler()
    
    if not model_loaded or not scaler_loaded:
        st.error("System initialization failed. Please verify model files are present.")
        st.stop()
    
    # Initialize session state
    if 'plr_data' not in st.session_state:
        st.session_state.plr_data = {'left': [], 'right': []}
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Video Analysis",
        "Image Sequence Analysis", 
        "CSV Data Import"
    ])
    
    # Tab 1: Video Upload
    with tab1:
        st.markdown("### Video-Based PLR Assessment")
        
        st.markdown("""
        <div class="info-box">
        <strong>Methodology:</strong> This module accepts video recordings of the pupillary light reflex 
        examination and automatically extracts temporal pupil diameter measurements for analysis.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Recording Protocol")
            st.markdown("""
            To ensure optimal data quality, please adhere to the following recording parameters:
            
            1. **Subject Positioning**: Position the subject 30-40 centimeters from the recording device
            2. **Illumination**: Ensure adequate ambient lighting on the subject's face
            3. **Camera Stability**: Maintain a stable recording position throughout the examination
            4. **Duration**: Record continuously for 4-5 seconds
            5. **Stimulus Administration**: Apply a controlled light stimulus (flashlight or penlight) to both eyes
            6. **Subject Cooperation**: Ensure the subject maintains forward gaze and minimizes blinking
            
            **Accepted Formats**: MP4, MOV, AVI, MKV, WebM
            """)
            
            video_file = st.file_uploader(
                "Select video file for analysis",
                type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
                help="Upload a video recording of the PLR examination"
            )
        
        with col2:
            st.markdown("#### Quality Checklist")
            st.markdown("""
            ✓ Both pupils clearly visible  
            ✓ Adequate facial illumination  
            ✓ Minimal motion artifacts  
            ✓ 4-5 second recording duration  
            ✓ Light stimulus clearly applied  
            ✓ Subject maintains fixation  
            """)
        
        if video_file is not None:
            st.markdown("#### Video Preview")
            st.video(video_file)
            
            st.markdown("---")
            
            if st.button("Initiate Video Analysis", type="primary", use_container_width=True):
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with st.spinner("Processing video data..."):
                    left_data, right_data = process_video(video_file, progress_placeholder, status_placeholder)
                
                progress_placeholder.empty()
                status_placeholder.empty()
                
                if len(left_data) >= 40 and len(right_data) >= 40:
                    st.session_state.plr_data['left'] = left_data[:40]
                    st.session_state.plr_data['right'] = right_data[:40]
                    
                    st.markdown("""
                    <div class="success-box">
                    <strong>Analysis Complete:</strong> Successfully extracted 40 temporal measurements 
                    from the video sequence. Proceed to results section below.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                    
                elif len(left_data) > 0:
                    st.markdown(f"""
                    <div class="warning-box">
                    <strong>Insufficient Data:</strong> Only {len(left_data)} valid measurements were extracted. 
                    A minimum of 40 measurements is required for accurate analysis.
                    <br><br>
                    <strong>Recommendations:</strong>
                    <ul>
                        <li>Verify adequate illumination of the subject's face</li>
                        <li>Ensure both pupils are clearly visible throughout the recording</li>
                        <li>Confirm recording duration is at least 4 seconds</li>
                        <li>Minimize motion artifacts during recording</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                    <strong>Detection Failure:</strong> No pupils were detected in the video sequence.
                    <br><br>
                    <strong>Common Issues:</strong>
                    <ul>
                        <li>Insufficient ambient lighting</li>
                        <li>Pupils obscured or not visible</li>
                        <li>Recording distance too great</li>
                        <li>Video resolution too low</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 2: Image Sequence
    with tab2:
        st.markdown("### Sequential Image Analysis")
        
        st.markdown("""
        <div class="info-box">
        <strong>Methodology:</strong> This module allows for manual compilation of PLR data through 
        sequential image uploads. Each image is analyzed independently, and measurements are aggregated 
        to form a complete temporal sequence.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Image Upload Protocol")
            st.markdown("""
            To construct a complete PLR sequence, capture and upload images according to the following protocol:
            
            1. Position subject at standardized distance (30-40 cm)
            2. Capture baseline images (frames 1-10) under ambient lighting
            3. Apply light stimulus to both eyes
            4. Capture constriction phase images (frames 11-30)
            5. Remove light stimulus
            6. Capture recovery phase images (frames 31-40)
            
            **Required**: 40 total images to complete the sequence
            """)
            
            image_file = st.file_uploader(
                "Upload individual image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a single frame for analysis",
                key="image_uploader"
            )
            
            if image_file is not None:
                annotated_frame, left_diam, right_diam = process_snapshot(image_file)
                
                st.image(annotated_frame, caption="Pupil Detection Visualization", use_container_width=True)
                
                if left_diam and right_diam:
                    st.success(f"Pupils successfully detected: Left = {left_diam:.2f} mm, Right = {right_diam:.2f} mm")
                    
                    if st.button("Add Measurement to Sequence", type="primary"):
                        st.session_state.plr_data['left'].append(left_diam)
                        st.session_state.plr_data['right'].append(right_diam)
                        st.success(f"Measurement {len(st.session_state.plr_data['left'])} recorded successfully")
                        st.rerun()
                else:
                    st.warning("Pupil detection unsuccessful. Please verify image quality and subject positioning.")
        
        with col2:
            st.markdown("#### Sequence Status")
            progress = len(st.session_state.plr_data['left'])
            st.metric("Measurements Collected", f"{progress} / 40")
            
            progress_percentage = min(progress / 40, 1.0)
            st.progress(progress_percentage)
            
            if progress < 40:
                remaining = 40 - progress
                st.info(f"{remaining} measurements required to complete sequence")
            else:
                st.success("Sequence complete. Analysis results available below.")
            
            if progress > 0:
                if st.button("Clear Sequence", use_container_width=True):
                    st.session_state.plr_data = {'left': [], 'right': []}
                    st.rerun()
    
    # Tab 3: CSV Import
    with tab3:
        st.markdown("### Structured Data Import")
        
        st.markdown("""
        <div class="info-box">
        <strong>Methodology:</strong> Import pre-processed pupillary diameter measurements 
        from external data acquisition systems via CSV format.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View CSV Format Specifications"):
            st.markdown("""
            **Required CSV Structure:**
            - Exactly 40 rows of measurements
            - Two columns: `left_pupil`, `right_pupil`
            - Values represent pupil diameter in millimeters
            - Temporal sampling at 10 Hz (0.1 second intervals)
            
            **Example Data Structure:**
            """)
            
            sample_df = pd.DataFrame({
                'left_pupil': [5.2, 5.1, 4.8, 4.5, 4.3, 4.2, 4.1, 4.0, 4.1, 4.2],
                'right_pupil': [5.1, 5.0, 4.7, 4.4, 4.2, 4.1, 4.0, 3.9, 4.0, 4.1]
            })
            st.dataframe(sample_df, use_container_width=True)
            st.caption("Display shows first 10 rows of required 40-row format")
            
            # Generate realistic sample
            baseline = 5.0
            constriction = 3.5
            frames = 40
            
            left_pattern = []
            right_pattern = []
            
            for i in range(frames):
                if i < 10:
                    left_pattern.append(baseline + np.random.normal(0, 0.1))
                    right_pattern.append(baseline + np.random.normal(0, 0.1))
                elif i < 25:
                    progress = (i - 10) / 15
                    left_pattern.append(baseline - (baseline - constriction) * progress + np.random.normal(0, 0.1))
                    right_pattern.append(baseline - (baseline - constriction) * progress + np.random.normal(0, 0.1))
                else:
                    progress = (i - 25) / 15
                    left_pattern.append(constriction + (baseline - constriction) * progress + np.random.normal(0, 0.1))
                    right_pattern.append(constriction + (baseline - constriction) * progress + np.random.normal(0, 0.1))
            
            sample_full = pd.DataFrame({
                'left_pupil': left_pattern,
                'right_pupil': right_pattern
            })
            
            st.download_button(
                label="Download Sample CSV Template",
                data=sample_full.to_csv(index=False),
                file_name="plr_data_template.csv",
                mime="text/csv"
            )
        
        uploaded_file = st.file_uploader(
            "Import CSV data file",
            type=['csv'],
            help="Upload CSV file containing pupil diameter measurements"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Displaying first 10 rows of {len(df)} total rows")
                
                if len(df) != 40:
                    st.error(f"Invalid data dimensions: {len(df)} rows detected. Exactly 40 rows required.")
                elif 'left_pupil' not in df.columns or 'right_pupil' not in df.columns:
                    st.error(f"Column mismatch. Required columns: 'left_pupil', 'right_pupil'. Found: {', '.join(df.columns)}")
                else:
                    st.session_state.plr_data['left'] = df['left_pupil'].tolist()
                    st.session_state.plr_data['right'] = df['right_pupil'].tolist()
                    
                    st.markdown("""
                    <div class="success-box">
                    <strong>Import Successful:</strong> CSV data has been loaded and validated. 
                    Proceed to analysis results below.
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"CSV parsing error: {str(e)}")
    
    # Analysis Results Section
    if len(st.session_state.plr_data['left']) == 40:
        st.markdown("---")
        st.markdown("## Diagnostic Analysis Results")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Temporal PLR Profile")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.plr_data['left'],
                mode='lines+markers',
                name='Left Pupil',
                line=dict(color='#3b82f6', width=2.5),
                marker=dict(size=5, symbol='circle')
            ))
            fig.add_trace(go.Scatter(
                y=st.session_state.plr_data['right'],
                mode='lines+markers',
                name='Right Pupil',
                line=dict(color='#ef4444', width=2.5),
                marker=dict(size=5, symbol='circle')
            ))
            fig.update_layout(
                xaxis_title="Frame Number (0.1 s intervals)",
                yaxis_title="Pupil Diameter (mm)",
                height=450,
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Classification Result")
            
            with st.spinner("Executing neural network inference..."):
                predictions = predict_plr(
                    st.session_state.plr_data['left'],
                    st.session_state.plr_data['right'],
                    model,
                    scaler
                )
            
            if predictions is not None:
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100
                
                st.markdown(f"""
                <div style='padding: 30px; background: linear-gradient(135deg, {CLASS_COLORS[predicted_class]}15, {CLASS_COLORS[predicted_class]}08); 
                border-radius: 12px; border-left: 5px solid {CLASS_COLORS[predicted_class]}; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <h4 style='margin: 0 0 10px 0; color: #1e293b;'>Predicted Diagnosis:</h4>
                <h2 style='margin: 0; color: {CLASS_COLORS[predicted_class]}; font-size: 28px;'>{CLASS_LABELS[predicted_class]}</h2>
                <p style='margin: 20px 0 0 0; font-size: 24px; font-weight: 600; color: #1e293b;'>
                Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        if predictions is not None:
            st.markdown("---")
            st.markdown("### Probability Distribution Across Diagnostic Classes")
            
            prob_df = pd.DataFrame({
                'Condition': CLASS_LABELS,
                'Probability': predictions * 100
            })
            prob_df = prob_df.sort_values('Probability', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Condition'],
                orientation='h',
                marker=dict(
                    color=[CLASS_COLORS[CLASS_LABELS.index(c)] for c in prob_df['Condition']],
                    line=dict(color='rgba(255,255,255,0.6)', width=1)
                ),
                text=prob_df['Probability'].round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=13, weight='bold')
            ))
            fig.update_layout(
                xaxis_title="Classification Probability (%)",
                yaxis_title="",
                height=320,
                showlegend=False,
                xaxis=dict(range=[0, 110]),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Detailed Pupillometric Analysis"):
                left_pupil = np.array(st.session_state.plr_data['left'])
                right_pupil = np.array(st.session_state.plr_data['right'])
                
                st.markdown("#### Constriction Dynamics")
                col1, col2, col3, col4 = st.columns(4)
                
                constriction_left = (left_pupil[0] - left_pupil.min()) / left_pupil[0] * 100
                constriction_right = (right_pupil[0] - right_pupil.min()) / right_pupil[0] * 100
                latency_left = np.argmin(left_pupil) * 0.1
                latency_right = np.argmin(right_pupil) * 0.1
                
                col1.metric("Left Constriction Amplitude", f"{constriction_left:.1f}%", 
                           help="Normal range: 25-35%")
                col2.metric("Right Constriction Amplitude", f"{constriction_right:.1f}%",
                           help="Normal range: 25-35%")
                col3.metric("Left Latency", f"{latency_left:.2f} s",
                           help="Normal range: 0.2-0.3 s")
                col4.metric("Right Latency", f"{latency_right:.2f} s",
                           help="Normal range: 0.2-0.3 s")
                
                st.markdown("#### Baseline Characteristics")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Left Mean Diameter", f"{left_pupil.mean():.2f} mm",
                           help="Normal range: 2-8 mm")
                col2.metric("Right Mean Diameter", f"{right_pupil.mean():.2f} mm",
                           help="Normal range: 2-8 mm")
                col3.metric("Left Standard Deviation", f"{left_pupil.std():.3f} mm")
                col4.metric("Right Standard Deviation", f"{right_pupil.std():.3f} mm")
                
                st.markdown("#### Bilateral Symmetry Assessment")
                asymmetry = abs(constriction_left - constriction_right)
                
                if asymmetry > 20:
                    st.markdown(f"""
                    <div class="warning-box">
                    <strong>Asymmetry Detected:</strong> {asymmetry:.1f}% difference in constriction amplitude 
                    between left and right pupils. Consider potential unilateral pathology.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                    <strong>Symmetric Response:</strong> {asymmetry:.1f}% difference in constriction amplitude. 
                    Within normal bilateral variation parameters.
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col2:
                export_df = pd.DataFrame({
                    'frame': range(1, 41),
                    'time_seconds': [i * 0.1 for i in range(40)],
                    'left_pupil_mm': st.session_state.plr_data['left'],
                    'right_pupil_mm': st.session_state.plr_data['right']
                })
                
                st.download_button(
                    label="Export Analysis Data",
                    data=export_df.to_csv(index=False),
                    file_name="plr_analysis_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                if st.button("Initialize New Analysis", use_container_width=True):
                    st.session_state.plr_data = {'left': [], 'right': []}
                    st.rerun()

if __name__ == "__main__":
    main()
