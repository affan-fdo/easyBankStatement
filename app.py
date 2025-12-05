import streamlit as st
import pdf2image
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
import shutil

# Load trained model
@st.cache_resource
def load_model():
    return YOLO('runs/detect/train3/weights/best.pt')

def pdf_to_images(pdf_file):
    """Convert PDF pages to images"""
    poppler_path = r"C:\Users\user\Desktop\bankStatementParser\poppler-25.07.0\Library\bin"
    images = pdf2image.convert_from_bytes(pdf_file.read(), dpi=200, poppler_path=poppler_path)
    return images

def clear_output_dir(output_dir):
    """Clear all files in output directory"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

def detect_and_save_tables(model, image, page_num, output_dir):
    """Detect tables and save cropped regions"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run YOLO detection
    results = model.predict(img_array, conf=0.5)
    
    saved_tables = []
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        
        # Crop table region
        table_crop = img_array[y1:y2, x1:x2]
        
        # Save cropped table
        table_filename = f"page_{page_num}_table_{i+1}.jpg"
        table_path = os.path.join(output_dir, table_filename)
        cv2.imwrite(table_path, cv2.cvtColor(table_crop, cv2.COLOR_RGB2BGR))
        saved_tables.append(table_filename)
    
    return saved_tables

def main():
    st.title("Bank Statement Table Detector")
    
    # Create output directory
    output_dir = "outputTemp"
    os.makedirs(output_dir, exist_ok=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        # Clear output directory on new upload
        clear_output_dir(output_dir)
        st.success("PDF uploaded successfully!")
        
        if st.button("Process PDF"):
            with st.spinner("Loading model..."):
                model = load_model()
            
            with st.spinner("Converting PDF to images..."):
                images = pdf_to_images(uploaded_file)
            
            st.info(f"Processing {len(images)} pages...")
            
            all_tables = []
            progress_bar = st.progress(0)
            
            for page_num, image in enumerate(images, 1):
                st.write(f"Processing page {page_num}...")
                
                # Detect and save tables
                tables = detect_and_save_tables(model, image, page_num, output_dir)
                all_tables.extend(tables)
                
                progress_bar.progress(page_num / len(images))
            
            st.success(f"Processing complete! Found {len(all_tables)} tables.")
            st.write(f"Tables saved in: {output_dir}")
            
            # Display saved table filenames
            if all_tables:
                st.write("Detected tables:")
                for table in all_tables:
                    st.write(f"- {table}")

if __name__ == "__main__":
    main()