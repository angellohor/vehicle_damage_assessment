
import streamlit as st
import requests
from PIL import Image
import json
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vehicle Damage",
    page_icon="🚗",
    layout="wide"
)

# --- MAIN TITLE & DESCRIPTION ---
st.title("🚗 AI Vehicle Damage Assessment ")
st.markdown("""
This application leverages **Deep Learning (YOLOv8)** to automatically detect and classify vehicle damages.
**Pipeline:**
1. **Part Segmentation:** Identifies vehicle body parts (High-Res Model).
2. **Damage Detection:** Identifies scratches and dents (SAHI Tiling Strategy).
3. **Severity Logic:** Classifies damages as Minor or Severe based on relative area.
""")

# --- SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.info("⚠️ Ensure the API (`python app.py`) is running on port 8000.")
    
    # API URL Input
    default_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
    
    api_url = st.text_input("API Endpoint", default_url)
    st.write("---")
    st.write("🛠️ **Tech Stack:**")
    st.code("Python 3.10\nYOLOv8 (Segmentation)\nFastAPI (Backend)\nStreamlit (Frontend)\nOpenCV (Processing)", language="text")
    
    st.write("---")
    st.markdown("Developed by **Ángel López Hortelano**")

# --- IMAGE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Upload a vehicle image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 1. Display Original Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, width='stretch')

    # ANALYSIS BUTTON
    if st.button("🔍 Analyze Damages", type="primary"):
        with st.spinner('Processing image with Neural Networks...'):
            try:
                # 2. Send image to API (Backend)
                # We reset the file pointer to the beginning
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # POST Request
                response = requests.post(api_url, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 3. Display Processed Image
                    with col2:
                        st.subheader("🤖 AI Analysis Output")
                        # Retrieve the path from the JSON response
                        result_path = data.get("imagen_procesada") 
                        
                        if result_path and os.path.exists(result_path):
                            st.image(result_path, width='stretch')
                        else:
                            st.warning("⚠️ Processed image not found. (Are Streamlit and API running on the same machine?)")

                    # 4. Display Technical Report
                    st.write("---")
                    st.subheader("📋 Technical Assessment Report")
                    
                    report = data.get("reporte_daños", {})
                    
                    if not report:
                        st.success("✅ No significant damages detected on visible parts.")
                    else:
                        # Iterate through parts and display alerts
                        for part, damages in report.items():
                            with st.expander(f"🔴 Zone: {part.upper()}", expanded=True):
                                for damage in damages:
                                    if "SEVERE" in damage:
                                        st.error(f"⚠️ {damage} - (Recommended Action: **Replacement**)")
                                    else:
                                        st.warning(f"🛠️ {damage} - (Recommended Action: **Repair/Paint**)")
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.write(response.text)
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection refused. Is the Backend running? Try: `python app.py`")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")