import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

# Thiết lập tiêu đề ứng dụng
st.title("Hand Writing Digit Recognition")

# Cấu hình canvas
st.header("Write a number")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Màu tô (đen - không sử dụng vì chỉ cho nét vẽ)
    stroke_width=10,                # Độ dày nét vẽ
    stroke_color="#000000",         # Màu nét vẽ (đen)
    background_color="#FFFFFF",     # Màu nền (trắng)
    width=280,                      # Chiều rộng (28x10)
    height=280,                     # Chiều cao (28x10)
    drawing_mode="freedraw",        # Chế độ vẽ tay
    key="canvas",                   # Khóa để định danh
)

# Xử lý kết quả sau khi vẽ

if canvas_result.image_data is not None:
    # Chuyển đổi ảnh thành kích thước 28x28
    image_array = np.array(canvas_result.image_data)
    
    # Lấy giá trị grayscale và chuyển thành nhị phân (0 hoặc 255)
    grayscale_image = np.mean(image_array[:, :, :3], axis=2)  # Chuyển thành grayscale
    binary_image = (grayscale_image < 128).astype(np.uint8) * 255  # Chuyển thành trắng-đen

    # Hiển thị hình ảnh 28x28
    st.subheader("Hình ảnh 28x28:")
    st.image(binary_image, width=200, clamp=True)

    # Hiển thị dữ liệu numpy
    st.subheader("Ma trận dữ liệu (28x28):")
    st.write(binary_image)


