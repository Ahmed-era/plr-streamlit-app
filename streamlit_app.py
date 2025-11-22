import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import cv2
from PIL import Image
import time

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
    # Assuming average pupil is 3-8mm and typical camera resolution
    pixel_to_mm = 0.1  # calibration factor
    left_diameter = left_pupil * 2 * pixel_to_mm if left_pupil else None
    right_diameter = right_pupil * 2 * pixel_to_mm if right_pupil else None
    
    return frame, left_diameter, right_diameter

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
    st.markdown("**Real-time Pupillary Light Reflex Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
        ### 📹 How to Record:
        1. **Position yourself** 30-40cm from camera
        2. **Ensure good lighting** on your face
        3. **Look straight** at the camera
        4. **Click "Start Recording"**
        5. **Shine a light** at your eyes when prompted
        6. **Hold still** for 4 seconds
        
        ### 🎯 Detects:
        - ✅ Normal
        - 🔴 RAPD
        - 🔵 Horner's Syndrome
        - 🟡 Glaucoma
        - 🟣 Compressive Neuropathy
        
        **Model Accuracy:** 93-97%
        """)
        
        st.warning("⚠️ This is a screening tool for educational purposes. Not a substitute for professional medical diagnosis.")
    
    # Load model
    model, scaler = load_model()
    
    # Main interface
    st.markdown("---")
    
    # Recording section
    st.subheader("📹 Pupil Recording")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        camera_input = st.camera_input("Position yourself and click to capture")
        
        if camera_input is not None:
            # Convert to OpenCV format
            image = Image.open(camera_input)
            frame = np.array(image)
            
            # Detect pupils
            annotated_frame, left_diam, right_diam = detect_pupils(frame)
            
            # Show annotated frame
            st.image(annotated_frame, caption="Pupil Detection", use_container_width=True)
            
            # Display detected sizes
            if left_diam and right_diam:
                st.success(f"✅ Pupils detected! Left: {left_diam:.2f}mm, Right: {right_diam:.2f}mm")
            else:
                st.warning("⚠️ Could not detect both pupils. Please adjust lighting and position.")
    
    with col2:
        st.markdown("### 🎬 Recording Controls")
        
        # Recording state
        if 'recording' not in st.session_state:
            st.session_state.recording = False
            st.session_state.plr_data = {'left': [], 'right': [], 'timestamps': []}
        
        # Start/Stop recording button
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                st.session_state.recording = True
                st.session_state.plr_data = {'left': [], 'right': [], 'timestamps': []}
                st.rerun()
        else:
            st.warning("🎥 Recording in progress...")
            if st.button("⏹️ Stop Recording", use_container_width=True):
                st.session_state.recording = False
                st.rerun()
        
        # Show recording status
        if st.session_state.recording:
            st.info(f"📊 Frames captured: {len(st.session_state.plr_data['left'])}/40")
            
            # Automatic stop after 40 frames
            if len(st.session_state.plr_data['left']) >= 40:
                st.session_state.recording = False
                st.success("✅ Recording complete!")
                st.rerun()
    
    # Manual data entry option
    st.markdown("---")
    st.subheader("📝 Or Upload CSV Data")
    
    uploaded_file = st.file_uploader(
        "Upload PLR CSV File (40 rows: left_pupil, right_pupil)",
        type=['csv']
    )
    
    # Process uploaded CSV
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if len(df) != 40:
                st.error("❌ CSV must have exactly 40 rows!")
            elif 'left_pupil' not in df.columns or 'right_pupil' not in df.columns:
                st.error("❌ CSV must have 'left_pupil' and 'right_pupil' columns!")
            else:
                # Store in session state
                st.session_state.plr_data['left'] = df['left_pupil'].tolist()
                st.session_state.plr_data['right'] = df['right_pupil'].tolist()
                st.success("✅ CSV data loaded successfully!")
    
    # Analysis section
    if len(st.session_state.plr_data['left']) == 40:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Show data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 PLR Trace")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.plr_data['left'],
                mode='lines+markers',
                name='Left Pupil',
                line=dict(color='#3498db', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=st.session_state.plr_data['right'],
                mode='lines+markers',
                name='Right Pupil',
                line=dict(color='#e74c3c', width=2)
            ))
            fig.update_layout(
                xaxis_title="Frame Number",
                yaxis_title="Diameter (mm)",
                height=300
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
                
                # Display prediction
                st.markdown("#### 🎯 Prediction")
                st.markdown(f"# {CLASS_LABELS[predicted_class]}")
                st.markdown(f"### Confidence: {confidence:.1f}%")
        
        # Probability distribution
        if predictions is not None:
            st.markdown("---")
            st.subheader("📊 Class Probabilities")
            
            prob_df = pd.DataFrame({
                'Condition': CLASS_LABELS,
                'Probability': predictions * 100
            })
            
            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Condition'],
                orientation='h',
                marker=dict(color=CLASS_COLORS)
            ))
            fig.update_layout(
                xaxis_title="Probability (%)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            with st.expander("🔍 Detailed Metrics"):
                left_pupil = np.array(st.session_state.plr_data['left'])
                right_pupil = np.array(st.session_state.plr_data['right'])
                
                col1, col2, col3, col4 = st.columns(4)
                
                constriction_left = (left_pupil[0] - left_pupil.min()) / left_pupil[0] * 100
                constriction_right = (right_pupil[0] - right_pupil.min()) / right_pupil[0] * 100
                
                col1.metric("Left Constriction", f"{constriction_left:.1f}%")
                col2.metric("Right Constriction", f"{constriction_right:.1f}%")
                col3.metric("Left Mean", f"{left_pupil.mean():.2f} mm")
                col4.metric("Right Mean", f"{right_pupil.mean():.2f} mm")
                
                # Download results
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=pd.DataFrame({
                        'left_pupil': left_pupil,
                        'right_pupil': right_pupil
                    }).to_csv(index=False),
                    file_name="plr_recording.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
