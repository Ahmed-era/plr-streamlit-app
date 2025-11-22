import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="PLR Neuropathy Detection",
    page_icon="🔬",
    layout="wide"
)

# Load model and scaler
# Load model and scaler
# Load model and scaler
@st.cache_resource
def load_model():
    try:
        # Load model without compilation
        model = tf.keras.models.load_model(
            'plr_model_glaucoma.h5',
            compile=False
        )
        # Manually compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please check that the model file is uploaded correctly.")
        st.stop()
    
    try:
        scaler = joblib.load('plr_scaler_glaucoma.pkl')
    except Exception as e:
        st.error(f"❌ Error loading scaler: {str(e)}")
        st.stop()
    
    return model, scaler

model, scaler = load_model()

# Class labels
CLASS_LABELS = ['Normal', 'RAPD', 'Horner\'s Syndrome', 'Glaucoma', 'Compressive Neuropathy']
CLASS_COLORS = ['#00cc66', '#ff6b6b', '#4ecdc4', '#ff9f40', '#9b59b6']

# Title
st.title("🔬 PLR Neuropathy Detection System")
st.markdown("**AI-powered optic neuropathy screening using Pupillary Light Reflex analysis**")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system analyzes pupillary light reflex patterns to detect:
    - ✅ Normal
    - 🔴 RAPD (Relative Afferent Pupillary Defect)
    - 🔵 Horner's Syndrome
    - 🟡 Glaucoma
    - 🟣 Compressive Neuropathy
    
    **Model Accuracy:** 93-97%
    """)
    
    st.header("📋 CSV Format")
    st.markdown("""
    Your CSV should have:
    - **40 rows** (4 seconds @ 10fps)
    - **2 columns:** `left_pupil`, `right_pupil`
    - Values in millimeters (mm)
    """)
    
    st.warning("⚠️ This is a screening tool for educational purposes. Not a substitute for professional medical diagnosis.")

# File upload
uploaded_file = st.file_uploader("Upload PLR CSV File", type=['csv'])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate
        if len(df) != 40:
            st.error("❌ CSV must have exactly 40 rows!")
            st.stop()
        
        if 'left_pupil' not in df.columns or 'right_pupil' not in df.columns:
            st.error("❌ CSV must have 'left_pupil' and 'right_pupil' columns!")
            st.stop()
        
        # Show data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Uploaded Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("📈 Pupil Diameter Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['left_pupil'],
                mode='lines',
                name='Left Pupil',
                line=dict(color='#3498db', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=df['right_pupil'],
                mode='lines',
                name='Right Pupil',
                line=dict(color='#e74c3c', width=2)
            ))
            fig.update_layout(
                xaxis_title="Frame Number",
                yaxis_title="Diameter (mm)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Process and predict
        with st.spinner('🔄 Analyzing PLR patterns...'):
            # Extract features
            left_pupil = df['left_pupil'].values
            right_pupil = df['right_pupil'].values
            
            # Prepare sequence
            sequence = np.column_stack([left_pupil, right_pupil])
            sequence = sequence.reshape(1, 40, 2)
            
            # Predict
            predictions = model.predict(sequence, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
        
        # Display results
        st.success("✅ Analysis Complete!")
        
        # Main prediction
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🎯 Predicted Condition")
            st.markdown(f"# {CLASS_LABELS[predicted_class]}")
        
        with col2:
            st.markdown(f"### 📊 Confidence")
            st.markdown(f"# {confidence:.1f}%")
        
        # Probability chart
        st.markdown("---")
        st.subheader("📊 Class Probabilities")
        
        prob_df = pd.DataFrame({
            'Condition': CLASS_LABELS,
            'Probability': predictions[0] * 100
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
            col1, col2, col3, col4 = st.columns(4)
            
            constriction_left = (left_pupil[0] - left_pupil.min()) / left_pupil[0] * 100
            constriction_right = (right_pupil[0] - right_pupil.min()) / right_pupil[0] * 100
            
            col1.metric("Left Constriction", f"{constriction_left:.1f}%")
            col2.metric("Right Constriction", f"{constriction_right:.1f}%")
            col3.metric("Left Mean Diameter", f"{left_pupil.mean():.2f} mm")
            col4.metric("Right Mean Diameter", f"{right_pupil.mean():.2f} mm")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    # Welcome message
    st.info("👆 Upload a CSV file to begin analysis")
    
    # Sample data format
    st.subheader("📋 Sample CSV Format")
    sample_df = pd.DataFrame({
        'left_pupil': [5.2, 5.1, 4.8, 4.5, 4.3],
        'right_pupil': [5.1, 5.0, 4.7, 4.4, 4.2]
    })
    st.dataframe(sample_df, use_container_width=True)
    st.caption("Your CSV should have 40 rows in this format")



