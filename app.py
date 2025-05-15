import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import functions as func

# Information dictionary about each editing option
INFO = {
    "Basic Adjustments": """
    **Basic Adjustments** allow you to modify fundamental properties of your image:
    - **Brightness**: Increase or decrease the overall luminosity
    - **Contrast**: Enhance or reduce the difference between light and dark areas
    - **Rotation**: Rotate the image by a specific angle in degrees
    - **Flip**: Mirror the image horizontally or vertically
    """,
    
    "Filters": """
    **Filters** apply different effects to transform your image:
    - **Grayscale**: Convert image to black and white
    - **Gaussian Blur**: Smooth image with a Gaussian function (good for reducing noise)
    - **Median Blur**: Replace each pixel with median of neighboring pixels (good for salt-and-pepper noise)
    - **Box Blur**: Simple averaging of surrounding pixels
    - **Sepia**: Apply a reddish-brown vintage effect
    - **Negative**: Invert all colors in the image
    - **Color Boosts**: Enhance specific color channels (Red, Green, or Blue)
    """,
    
    "Edge Detection": """
    **Canny Edge Detection** identifies boundaries of objects within images:
    - **Low Threshold**: Determines which weak edges are included
    - **High Threshold**: Determines which edges are definitely included
    
    The algorithm detects strong edges (above high threshold) and includes connected edges between thresholds.
    """,
    
    "Thresholding": """
    **Thresholding** segments an image by setting pixel values based on a threshold:
    - **Binary**: Pixels above threshold become white, below become black
    - **Binary Inverted**: Reverses Binary thresholding
    - **Truncate**: Pixels above threshold are set to the threshold value
    - **To Zero**: Pixels below threshold become black, others unchanged
    - **To Zero Inverted**: Reverses To Zero thresholding
    - **Adaptive Methods**: Threshold value is calculated for smaller regions (better for variable lighting)
    - **Otsu's Method**: Automatically determines optimal threshold value
    """,
    
    "Face Detection": """
    **Face Detection** uses Haar Cascade classifiers to identify human faces in images.
    The algorithm scans the image at different scales and identifies areas that match facial patterns.
    Detected faces are outlined with blue rectangles.
    """,
    
    "Morphological Operations": """
    **Morphological Operations** process images based on shapes:
    - **Erosion**: Shrinks bright regions, expands dark regions
    - **Dilation**: Expands bright regions, shrinks dark regions
    - **Opening**: Erosion followed by dilation (removes small bright spots)
    - **Closing**: Dilation followed by erosion (fills small dark holes)
    - **Gradient**: Difference between dilation and erosion (outlines objects)
    
    Kernel Size determines how aggressively the operation is applied.
    """,
    
    "Contour Detection": """
    **Contour Detection** finds continuous curves along boundaries:
    - **Threshold 1 & 2**: Control the edge detection sensitivity
    
    The algorithm uses Canny edge detection to find edges, then identifies and draws contours in green.
    Useful for shape analysis, object detection, and recognition.
    """,
    
    "Crop Image": """
    **Crop Image** allows you to select a specific portion of your image by setting:
    - **X Start**: Left edge of the crop area
    - **Y Start**: Top edge of the crop area
    - **X End**: Right edge of the crop area
    - **Y End**: Bottom edge of the crop area
    
    The preview shows the selected region. Adjust the sliders to perfect your crop.
    """
}

def main():
    st.set_page_config(page_title="Image Editor", layout="wide")
    st.title("Image Editor App")
    
    # Help information at the top
    with st.expander("How to use this app"):
        st.markdown("""
        ### Welcome to Image Editor App!
        
        **Quick Start Guide:**
        1. Upload an image using the file uploader in the sidebar
        2. Select an editing option from the dropdown menu
        3. Adjust parameters to modify your image
        4. Download the edited image using the button at the bottom
        
        Each editing tool has an information button that explains its functionality in detail.
        
        **Tips:**
        - Try combining different edits by downloading and re-uploading the image
        - For best results with face detection, ensure faces are clearly visible
        - Edge detection works well on images with clear boundaries
        """)
    
    # Sidebar options
    st.sidebar.title("Tools")
    
    # Upload image
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # About section in sidebar
    with st.sidebar.expander("About"):
        st.markdown("""
        **Image Editor App**
        
        This application provides various image processing tools through an easy-to-use interface.
        
        Features:
        - Basic image adjustments
        - Filters and effects
        - Edge and contour detection
        - Face detection
        - Thresholding
        - Morphological operations
        
        Built with Streamlit and OpenCV.
        """)
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        
        # Create a copy of the original image for editing
        edited_image = image.copy()
        
        # Select editing option
        edit_option = st.sidebar.selectbox(
            "Select an editing option",
            ["Basic Adjustments", "Filters", "Edge Detection", "Thresholding", 
             "Face Detection", "Morphological Operations", "Contour Detection", "Crop Image"]
        )
        
        # Information button
        with st.sidebar.expander("ℹ️ Information about this tool"):
            st.markdown(INFO[edit_option])
        
        # Basic adjustments
        if edit_option == "Basic Adjustments":
            st.subheader("Basic Adjustments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                brightness = st.slider("Brightness", 0.0, 3.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.0, 3.0, 1.0, 0.1)
            
            with col2:
                rotation = st.slider("Rotation", -180, 180, 0, 5)
                flip_option = st.selectbox("Flip", ["None", "Horizontal", "Vertical"])
            
            edited_image = func.adjust_brightness(edited_image, brightness)
            edited_image = func.adjust_contrast(edited_image, contrast)
            
            if rotation != 0:
                edited_image = func.rotate_image(edited_image, rotation)
            
            if flip_option == "Horizontal":
                edited_image = func.flip_image(edited_image, 1)
            elif flip_option == "Vertical":
                edited_image = func.flip_image(edited_image, 0)
        
        # Filters
        elif edit_option == "Filters":
            st.subheader("Filters")
            
            filter_type = st.selectbox(
                "Select a filter",
                ["Grayscale", "Gaussian Blur", "Median Blur", "Box Blur", 
                 "Sepia", "Negative", "Blue Boost", "Red Boost", "Green Boost"]
            )
            
            if filter_type == "Grayscale":
                gray = func.convert_to_grayscale(edited_image)
                edited_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            elif filter_type in ["Gaussian Blur", "Median Blur", "Box Blur"]:
                kernel_size = st.slider("Kernel Size", 3, 25, 5, step=2)
                edited_image = func.apply_blur(edited_image, filter_type, kernel_size)
            
            else:  # Color filters
                edited_image = func.apply_color_filter(edited_image, filter_type)
        
        # Edge Detection
        elif edit_option == "Edge Detection":
            st.subheader("Canny Edge Detection")
            
            low_threshold = st.slider("Low Threshold", 0, 255, 100)
            high_threshold = st.slider("High Threshold", 0, 255, 200)
            
            edges = func.apply_canny_edge(edited_image, low_threshold, high_threshold)
            edited_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Thresholding
        elif edit_option == "Thresholding":
            st.subheader("Thresholding")
            
            threshold_type = st.selectbox(
                "Select threshold type",
                ["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted",
                 "Adaptive Mean", "Adaptive Gaussian", "Otsu's Binarization"]
            )
            
            if threshold_type not in ["Adaptive Mean", "Adaptive Gaussian", "Otsu's Binarization"]:
                threshold_value = st.slider("Threshold Value", 0, 255, 127)
            else:
                threshold_value = 127  # Not used for adaptive/Otsu methods
            
            max_value = st.slider("Max Value", 0, 255, 255)
            
            edited_image = func.apply_threshold(edited_image, threshold_value, max_value, threshold_type)
        
        # Face Detection
        elif edit_option == "Face Detection":
            st.subheader("Face Detection")
            
            edited_image, num_faces = func.detect_faces(edited_image)
            st.write(f"Detected {num_faces} face(s)")
        
        # Morphological Operations
        elif edit_option == "Morphological Operations":
            st.subheader("Morphological Operations")
            
            op_type = st.selectbox(
                "Select operation",
                ["Erosion", "Dilation", "Opening", "Closing", "Gradient"]
            )
            
            kernel_size = st.slider("Kernel Size", 1, 15, 5, step=2)
            
            edited_image = func.apply_morphological_op(edited_image, op_type, kernel_size)
        
        # Contour Detection
        elif edit_option == "Contour Detection":
            st.subheader("Contour Detection")
            
            threshold1 = st.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.slider("Threshold 2", 0, 255, 200)
            
            edited_image = func.detect_contours(edited_image, threshold1, threshold2)
        
        # Crop Image
        elif edit_option == "Crop Image":
            st.subheader("Crop Image")
            
            # Get image dimensions
            height, width = edited_image.shape[:2]
            
            # Create cropping controls with columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                x_start = st.slider("X Start", 0, width-10, 0)
                y_start = st.slider("Y Start", 0, height-10, 0)
            
            with col2:
                x_end = st.slider("X End", x_start+10, width, width)
                y_end = st.slider("Y End", y_start+10, height, height)
            
            # Draw rectangle on image to show crop area
            preview_img = edited_image.copy()
            cv2.rectangle(preview_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            
            # Show preview with rectangle
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="Crop Preview", use_container_width=True)
            
            # Apply crop when button is clicked
            if st.button("Apply Crop"):
                edited_image = func.crop_image(edited_image, x_start, y_start, x_end, y_end)
                # Update preview
                st.image(cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB), caption="Cropped Image", use_container_width=True)
        
        # Display the edited image
        if edit_option != "Crop Image":  # Don't show duplicate image for crop option
            st.image(cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB), caption="Edited Image", use_container_width=True)
        
        # Additional information about the current edit
        with st.expander("Technical Details"):
            st.write(f"Current editing mode: **{edit_option}**")
            
            # Show image dimensions
            height, width = edited_image.shape[:2]
            st.write(f"Image dimensions: {width} x {height} pixels")
            
            # Show memory usage
            memory_usage = edited_image.nbytes / (1024 * 1024)  # Convert to MB
            st.write(f"Memory usage: {memory_usage:.2f} MB")
            
            if edit_option == "Face Detection":
                _, num_faces = func.detect_faces(image)
                st.write(f"Number of faces detected: {num_faces}")
        
        # Download button for edited image
        if st.button("Download Edited Image"):
            edited_img_rgb = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(edited_img_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="edited_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()