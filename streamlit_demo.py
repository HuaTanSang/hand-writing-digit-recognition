import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2 
import torch 
import model_directory
import os

from LeNet import LeNet
from GoogLeNet import GoogleNet
from MLP import One_Layer_MLP, Three_Layer_MLP
from ResNet18 import ResNet18

def create_area_for_drawing() -> np.ndarray: 
    # Tạo ô để vẽ 
    draw_area = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  
        stroke_width=10,                
        stroke_color="#000000",         
        background_color="#FFFFFF",     
        width=280,                     
        height=280,                     
        drawing_mode="freedraw",        
        key="canvas",                   
    )

    binary_image = None 

    # Xử lý kết quả sau khi vẽ
    if draw_area.image_data is not None:
        image_array = np.array(draw_area.image_data)
        grayscale_image = np.mean(image_array[:, :, :3], axis=2) 
        binary_image = (grayscale_image < 128).astype(np.uint8) * 255  

    return binary_image

def preprocess_input(binary_image: np.ndarray, target_size=(28,28)) -> torch.Tensor: 
    if binary_image is None:
        return None
        
    # Resize image 
    resized_img = cv2.resize(binary_image, target_size, interpolation=cv2.INTER_AREA)

    # Convert to Tensor and normalize
    tensor_img = torch.Tensor(resized_img).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor_img

def predict_output(model: torch.nn.Module, tensor_img: torch.Tensor, is_mlp: bool = False) -> tuple:
    if tensor_img is None:
        return None, None
    
    # Flatten input for MLP models
    if is_mlp:
        tensor_img = tensor_img.view(-1, 28*28)
        
    model.eval()
    with torch.no_grad():
        predictions, _ = model(tensor_img, torch.tensor([0]))  # dummy label
        probabilities = torch.softmax(predictions, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence



def main(): 
    st.title("Hand Writing Digit Recognition")
    
    # Tạo hai cột
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Draw a number")
        binary_image = create_area_for_drawing()
        
        if binary_image is not None:
            st.image(binary_image, width=200, caption="Processed Image", clamp=True)
    
    # Load các model
    @st.cache_resource
    def load_models():
        lenet = LeNet() 
        lenet.load_state_dict(torch.load(model_directory.best_LeNet[3]))
        
        googlenet = GoogleNet() 
        googlenet.load_state_dict(torch.load(model_directory.best_GoogLeNet[3]))

        one_layer_mlp = One_Layer_MLP() 
        one_layer_mlp.load_state_dict(torch.load(model_directory.best_1_Layer_MLP[3]))

        three_layer_mlp = Three_Layer_MLP() 
        three_layer_mlp.load_state_dict(torch.load(model_directory.best_3_Layer_MLP[3]))

        resnet18 = ResNet18(in_dim=1, hidden_size=64)
        resnet18.load_state_dict(torch.load(model_directory.best_ResNet[3]))

        return lenet, googlenet, one_layer_mlp, three_layer_mlp, resnet18
    
    try:
        lenet, googlenet, one_layer_mlp, three_layer_mlp, resnet18 = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    input_img = preprocess_input(binary_image)
    
    # Hiển thị kết quả trong cột 2
    with col2:
        if input_img is not None:
            st.subheader("Predictions:")
            
            # Tạo một DataFrame để hiển thị kết quả
            results = []
            
            # LeNet
            digit, conf = predict_output(lenet, input_img)
            results.append({"Model": "LeNet", "Prediction": digit, "Confidence": f"{conf:.2%}"})
            
            # GoogLeNet
            digit, conf = predict_output(googlenet, input_img)
            results.append({"Model": "GoogLeNet", "Prediction": digit, "Confidence": f"{conf:.2%}"})
            
            # 1-Layer MLP
            digit, conf = predict_output(one_layer_mlp, input_img, is_mlp=True)
            results.append({"Model": "1-Layer MLP", "Prediction": digit, "Confidence": f"{conf:.2%}"})
            
            # 3-Layer MLP
            digit, conf = predict_output(three_layer_mlp, input_img, is_mlp=True)
            results.append({"Model": "3-Layer MLP", "Prediction": digit, "Confidence": f"{conf:.2%}"})
            
            # ResNet18
            digit, conf = predict_output(resnet18, input_img)
            results.append({"Model": "ResNet18", "Prediction": digit, "Confidence": f"{conf:.2%}"})
            
            # Hiển thị kết quả dưới dạng bảng
            st.table(results)
    
    # Thêm nút Clear để xóa vùng vẽ
    if st.button('Clear Drawing'):
        st.session_state['canvas'] = None
        st.experimental_rerun()

if __name__ == "__main__": 
    main()