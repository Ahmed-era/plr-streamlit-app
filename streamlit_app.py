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
    page_title="PLR Neuropathy Detection",
    page_icon="🔬",
    layout="wide"
)

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
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    try:
        scaler = joblib.load('plr_scaler_glaucoma.pkl')
    except Exception as e:
        st.error(f"❌ Error loading scaler: {str(e)}")
        st.stop()
    
    return model, scaler

# Pupil detection function
def detect_pupils(frame):
    """Detect pupils using OpenCV"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Detect circles (pupils)
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
        # Sort by x-coordinate (left to right)
        circles_sorted = sorted(circles[0, :], key=lambda x: x[0])
        
        if len(circles_sorted) >= 2:
            # Left pupil (smaller x)
            left_pupil = circles_sorted[0][2]  # radius
            # Right pupil (larger x)
            right_pupil = circles_sorted[1][2]  # radius
            
            # Draw circles on frame
            for circle in circles_sorted[:2]:
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    
    # Convert radius to diameter (mm - approximate)
    pixel_to_mm = 0.1  # calibration factor
    left_diameter = left_pupil * 2 * pixel_to_mm if left_pupil else None
    right_diameter = right_pupil * 2 * pixel_to_mm if right_pupil else None
    
    return frame, left_diameter, right_diameter

# Process video file
def process_video(video_file):
    """Extract pupil measurements from video"""
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    # Open video
    cap = cv2.VideoCapture(tfile.name)
    
    left_measurements = []
    right_measurements = []
    frames_processed = 0
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate frame skip to get ~40 frames evenly distributed
    frame_skip = max(1, total_frames // 40)
    frame_count = 0
    
    while frames_processed < 40 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_count % frame_skip == 0:
            # Detect pupils
            _, left_diam, right_diam = detect_pupils(frame)
            
            if left_diam and right_diam:
                left_measurements.append(left_diam)
                right_measurements.append(right_diam)
                frames_processed += 1
                
                # Update progress
                progress_bar.progress(min(frames_processed / 40, 1.0))
                status_text.text(f"Processing: {frames_processed}/40 frames")
        
        frame_count += 1
    
    cap.release()
    os.unlink(tfile.name)
    progress_bar.empty()
    status_text.empty()
    
    return left_measurements, right_measurements

# Process image snapshot
def process_snapshot(image_file):
    """Extract pupil measurements from single snapshot"""
    # Convert to OpenCV format
    image = Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Detect pupils
    annotated_frame, left_diam, right_diam = detect_pupils(frame)
    
    # Convert back to RGB for display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, left_diam, right_diam

# Prediction function
def predict_plr(left_data, right_data, model, scaler):
    """Analyze PLR data and predict condition"""
    try:
        left_pupil = np.array(left_data)
        right_pupil = np.array(right_data)
        
        # Prepare sequence for model
        sequence = np.column_stack([left_pupil, right_pupil])
        sequence = sequence.reshape(1, 40, 2)
        
        # Predict
        predictions = model.predict(sequence, verbose=0)
        
        return predictions[0]
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Main app
def main():
    CLASS_LABELS = ['Normal', 'RAPD', 'Horner\'s Syndrome', 'Glaucoma', 'Compressive Neuropathy']
    CLASS_COLORS = ['#00cc66', '#ff6b6b', '#4ecdc4', '#ff9f40', '#9b59b6']
    
    # Title
    st.title("🔬 PLR Neuropathy Detection System")
    st.markdown("**AI-powered Pupillary Light Reflex Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
        ### 📹 How to Use:
        
        **🎥 Recommended: Upload Video**
        1. Record 4-5 sec video on phone
        2. Show both eyes clearly
        3. Shine light during recording
        4. Upload and analyze
        
        **📸 Alternative: Take Snapshots**
        1. Take multiple photos (40)
        2. Upload photos one by one
        3. System collects measurements
        
        **📊 Or Upload CSV**
        - Pre-recorded data
        - 40 rows required
        
        ### 🎯 Detects:
        - ✅ Normal
        - 🔴 RAPD
        - 🔵 Horner's Syndrome
        - 🟡 Glaucoma
        - 🟣 Compressive Neuropathy
        
        **Model Accuracy:** 93-97%
        """)
        
        st.warning("⚠️ Educational purposes only.")
    
    # Load model
    model, scaler = load_model()
    
    # Initialize session state
    if 'plr_data' not in st.session_state:
        st.session_state.plr_data = {'left': [], 'right': []}
    
    # Main interface
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Upload Video (Recommended)", "📸 Take Snapshots", "📝 Upload CSV"])
    
    # Tab 1: Video Upload (PRIMARY METHOD)
    with tab1:
        st.subheader("🎥 Upload PLR Video")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 **Best method:** Record a 4-5 second video on your phone showing the pupillary light reflex test")
            
            st.markdown("""
            ### 📱 How to Record:
            1. **Open phone camera** (use rear camera for better quality)
            2. **Position** someone 30-40cm away
            3. **Start recording** and ensure both eyes are visible
            4. **Shine a flashlight** at the eyes (from another device/light source)
            5. **Record for 4-5 seconds** total
            6. **Stop recording**
            7. **Upload the video** below
            """)
            
            video_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
                help="Upload a 4-5 second video showing PLR test"
            )
        
        with col2:
            st.markdown("### ✅ Recording Tips:")
            st.success("✓ Good lighting on face")
            st.success("✓ Both eyes visible")
            st.success("✓ Hold camera steady")
            st.success("✓ 4-5 seconds long")
            st.success("✓ Shine bright light")
            st.success("✓ Subject looks at camera")
        
        if video_file is not None:
            # Show video preview
            st.video(video_file)
            
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                with st.spinner("🎬 Processing video frames..."):
                    left_data, right_data = process_video(video_file)
                
                if len(left_data) >= 40 and len(right_data) >= 40:
                    # Store first 40 measurements
                    st.session_state.plr_data['left'] = left_data[:40]
                    st.session_state.plr_data['right'] = right_data[:40]
                    st.success(f"✅ Successfully extracted 40 frames!")
                    st.balloons()
                    st.rerun()
                elif len(left_data) > 0:
                    st.warning(f"⚠️ Only detected {len(left_data)} valid frames. Need at least 40.")
                    st.info("💡 **Tips to improve detection:**\n- Ensure better lighting on the face\n- Make sure both eyes are clearly visible\n- Record for at least 4-5 seconds\n- Keep the camera steady")
                else:
                    st.error("❌ Could not detect any pupils in the video.")
                    st.info("💡 **Common issues:**\n- Video too dark\n- Eyes not visible\n- Camera too far away\n- Poor video quality")
    
    # Tab 2: Snapshot Method
    with tab2:
        st.subheader("📸 Take Multiple Snapshots")
        st.info("💡 Take 40 photos while performing the PLR test. Upload each photo individually.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Single image upload
            image_file = st.file_uploader(
                "Upload a snapshot",
                type=['jpg', 'jpeg', 'png'],
                help="Take a photo and upload it",
                key="snapshot_uploader"
            )
            
            if image_file is not None:
                # Process snapshot
                annotated_frame, left_diam, right_diam = process_snapshot(image_file)
                
                # Show annotated image
                st.image(annotated_frame, caption="Pupil Detection", use_container_width=True)
                
                # Display detection status
                if left_diam and right_diam:
                    st.success(f"✅ Pupils detected! Left: {left_diam:.2f}mm, Right: {right_diam:.2f}mm")
                    
                    # Add to collection button
                    if st.button("➕ Add This Snapshot to Collection", type="primary"):
                        st.session_state.plr_data['left'].append(left_diam)
                        st.session_state.plr_data['right'].append(right_diam)
                        st.success(f"✅ Snapshot {len(st.session_state.plr_data['left'])} added!")
                        st.rerun()
                else:
                    st.warning("⚠️ Could not detect both pupils. Please adjust lighting and retake photo.")
        
        with col2:
            st.markdown("### 📊 Collection Status")
            progress = len(st.session_state.plr_data['left'])
            st.metric("Snapshots Collected", f"{progress}/40")
            
            # Progress bar
            progress_percentage = min(progress / 40, 1.0)
            st.progress(progress_percentage)
            
            if progress < 40:
                st.info(f"Need {40 - progress} more snapshots")
            else:
                st.success("✅ Collection complete! Scroll down for results.")
            
            # Reset button
            if progress > 0:
                if st.button("🔄 Clear Collection", use_container_width=True):
                    st.session_state.plr_data = {'left': [], 'right': []}
                    st.rerun()
    
    # Tab 3: CSV Upload
    with tab3:
        st.subheader("📝 Upload CSV Data")
        
        # Sample CSV format
        with st.expander("📋 View Sample CSV Format"):
            sample_df = pd.DataFrame({
                'left_pupil': [5.2, 5.1, 4.8, 4.5, 4.3, 4.2, 4.1, 4.0, 4.1, 4.2],
                'right_pupil': [5.1, 5.0, 4.7, 4.4, 4.2, 4.1, 4.0, 3.9, 4.0, 4.1]
            })
            st.dataframe(sample_df, use_container_width=True)
            st.caption("Your CSV should have 40 rows (showing first 10)")
            
            # Create sample CSV with realistic PLR pattern
            baseline = 5.0
            constriction = 3.5
            frames = 40
            
            # Simulate PLR: baseline -> constrict -> recover
            left_pattern = []
            right_pattern = []
            
            for i in range(frames):
                if i < 10:  # Baseline
                    left_pattern.append(baseline + np.random.normal(0, 0.1))
                    right_pattern.append(baseline + np.random.normal(0, 0.1))
                elif i < 25:  # Constriction
                    progress = (i - 10) / 15
                    left_pattern.append(baseline - (baseline - constriction) * progress + np.random.normal(0, 0.1))
                    right_pattern.append(baseline - (baseline - constriction) * progress + np.random.normal(0, 0.1))
                else:  # Recovery
                    progress = (i - 25) / 15
                    left_pattern.append(constriction + (baseline - constriction) * progress + np.random.normal(0, 0.1))
                    right_pattern.append(constriction + (baseline - constriction) * progress + np.random.normal(0, 0.1))
            
            sample_full = pd.DataFrame({
                'left_pupil': left_pattern,
                'right_pupil': right_pattern
            })
            
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=sample_full.to_csv(index=False),
                file_name="plr_template.csv",
                mime="text/csv"
            )
        
        uploaded_file = st.file_uploader(
            "Upload PLR CSV File",
            type=['csv'],
            help="CSV with 40 rows: left_pupil, right_pupil columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Show preview
                st.markdown("**Preview of uploaded data:**")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(df)} rows")
                
                if len(df) != 40:
                    st.error(f"❌ CSV has {len(df)} rows but must have exactly 40 rows!")
                elif 'left_pupil' not in df.columns or 'right_pupil' not in df.columns:
                    st.error("❌ CSV must have 'left_pupil' and 'right_pupil' columns!")
                    st.info(f"Found columns: {', '.join(df.columns)}")
                else:
                    # Store in session state
                    st.session_state.plr_data['left'] = df['left_pupil'].tolist()
                    st.session_state.plr_data['right'] = df['right_pupil'].tolist()
                    st.success("✅ CSV data loaded successfully!")
                    st.balloons()
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    # Analysis section (shown if data exists)
    if len(st.session_state.plr_data['left']) == 40:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Show data visualization
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 📈 PLR Trace")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.plr_data['left'],
                mode='lines+markers',
                name='Left Pupil',
                line=dict(color='#3498db', width=3),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                y=st.session_state.plr_data['right'],
                mode='lines+markers',
                name='Right Pupil',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=6)
            ))
            fig.update_layout(
                xaxis_title="Frame Number (0.1s intervals)",
                yaxis_title="Pupil Diameter (mm)",
                height=400,
                hovermode='x unified',
                legend=dict(x=0.7, y=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Run prediction
            with st.spinner('🔄 Analyzing PLR patterns...'):
                predictions = predict_plr(
                    st.session_state.plr_data['left'],
                    st.session_state.plr_data['right'],
                    model,
                    scaler
                )
            
            if predictions is not None:
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100
                
                # Display prediction with styling
                st.markdown("#### 🎯 Diagnosis")
                st.markdown(
                    f"<div style='padding: 25px; background: linear-gradient(135deg, {CLASS_COLORS[predicted_class]}22, {CLASS_COLORS[predicted_class]}11); "
                    f"border-radius: 15px; border-left: 6px solid {CLASS_COLORS[predicted_class]}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>"
                    f"<h2 style='margin: 0; color: {CLASS_COLORS[predicted_class]}; font-size: 32px;'>{CLASS_LABELS[predicted_class]}</h2>"
                    f"<p style='margin: 15px 0 0 0; font-size: 28px; font-weight: bold;'>Confidence: {confidence:.1f}%</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        
        # Probability distribution
        if predictions is not None:
            st.markdown("---")
            st.subheader("📊 Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Condition': CLASS_LABELS,
                'Probability': predictions * 100
            })
            
            # Sort by probability
            prob_df = prob_df.sort_values('Probability', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Condition'],
                orientation='h',
                marker=dict(
                    color=[CLASS_COLORS[CLASS_LABELS.index(c)] for c in prob_df['Condition']],
                    line=dict(color='white', width=2)
                ),
                text=prob_df['Probability'].round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=14, weight='bold')
            ))
            fig.update_layout(
                xaxis_title="Probability (%)",
                height=300,
                showlegend=False,
                xaxis=dict(range=[0, 110])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            with st.expander("🔍 View Detailed Clinical Metrics"):
                left_pupil = np.array(st.session_state.plr_data['left'])
                right_pupil = np.array(st.session_state.plr_data['right'])
                
                st.markdown("### Pupil Constriction Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                constriction_left = (left_pupil[0] - left_pupil.min()) / left_pupil[0] * 100
                constriction_right = (right_pupil[0] - right_pupil.min()) / right_pupil[0] * 100
                latency_left = np.argmin(left_pupil) * 0.1  # Convert to seconds
                latency_right = np.argmin(right_pupil) * 0.1
                
                col1.metric("Left Constriction", f"{constriction_left:.1f}%", 
                           help="Normal: 25-35%")
                col2.metric("Right Constriction", f"{constriction_right:.1f}%",
                           help="Normal: 25-35%")
                col3.metric("Left Latency", f"{latency_left:.2f}s",
                           help="Normal: 0.2-0.3s")
                col4.metric("Right Latency", f"{latency_right:.2f}s",
                           help="Normal: 0.2-0.3s")
                
                st.markdown("### Baseline Measurements")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Left Mean", f"{left_pupil.mean():.2f} mm",
                           help="Normal: 2-8mm")
                col2.metric("Right Mean", f"{right_pupil.mean():.2f} mm",
                           help="Normal: 2-8mm")
                col3.metric("Left Variability", f"{left_pupil.std():.3f} mm")
                col4.metric("Right Variability", f"{right_pupil.std():.3f} mm")
                
                # Asymmetry check
                asymmetry = abs(constriction_left - constriction_right)
                st.markdown("### Symmetry Analysis")
                if asymmetry > 20:
                    st.warning(f"⚠️ Significant asymmetry detected: {asymmetry:.1f}% difference")
                else:
                    st.success(f"✅ Symmetric response: {asymmetry:.1f}% difference")
            
            # Download results
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col2:
                # Export data
                export_df = pd.DataFrame({
                    'frame': range(1, 41),
                    'time_seconds': [i * 0.1 for i in range(40)],
                    'left_pupil_mm': st.session_state.plr_data['left'],
                    'right_pupil_mm': st.session_state.plr_data['right']
                })
                
                st.download_button(
                    label="📥 Download Data CSV",
                    data=export_df.to_csv(index=False),
                    file_name="plr_analysis_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Clear data button
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.plr_data = {'left': [], 'right': []}
                    st.rerun()

if __name__ == "__main__":
    main()

