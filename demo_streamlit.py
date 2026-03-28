import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using menu
# st.title("Trung Tâm Tin Học")
st.image("banner_nhatot.png", use_column_width=True)

menu = ["Home", "Capstone Project", "Sử dụng các điều khiển", 
        "Gợi ý điều khiển project 1", "Gợi ý điều khiển project 2"]

choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
    st.write('''
    ### Chào mừng bạn đến với khóa học 
    ##### Đồ án TN''')

elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 2 chủ đề trong khóa học:    
    - Topic 1: Dự đoán giá nhà, phát hiện tin đăng bán nhà bất thường
    - Topic 2: Hệ thống gợi ý nhà dựa trên nội dung, phân cụm nhà
             """)

elif choice == 'Sử dụng các điều khiển':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)
    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")
    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")
    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive", "Busy"))
    st.write("You are", status)
    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)
    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)
    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)    
    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)
    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)
    # 11. Display JSON
    st.subheader("11. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)
    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)
          
elif choice == 'Gợi ý điều khiển project 1':
    st.write("##### Gợi ý điều khiển project 1: Dự đoán giá nhà và phát hiện tin đăng bán nhà bất thường")
    st.write("##### Dữ liệu mẫu")
    # đọc dữ liệu từ file subset_100motobykes.csv
    df = pd.read_csv("house_samples.csv")
    st.dataframe(df.head())   

    # Trường hợp 2: Đọc dữ liệu từ file csv, excel do người dùng tải lên
    st.write("### Đọc dữ liệu từ file csv do người dùng tải lên")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])   
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dữ liệu đã nhập:")
        st.dataframe(df.head())
    st.write("### 1. Dự đoán giá nhà")
    # Tạo điều khiển để người dùng nhập các thông tin về nhà
    # nhạp tieu de, mo ta, dia chi, gia, dien tich, so phong ngu, so phong tam, nam xay dung
    tieu_de = st.text_input("Nhập tiêu đề nhà")
    mo_ta = st.text_area("Nhập mô tả nhà")
    dia_chi = st.text_input("Nhập địa chỉ nhà")
    gia = st.slider("Nhập giá nhà (VND)", 0, 100000000000, 500000000, step=10000000)

    dien_tich = st.number_input("Nhập diện tích (m2)", min_value=0, step=1)
    so_phong_ngu = st.number_input("Nhập số phòng ngủ", min_value=0, step=1)
    so_phong_tam = st.number_input("Nhập số phòng tắm", min_value=0, step=1)
    so_tang = st.slider("Chọn số tầng", 0, 10, 1, step=1)
 
    du_doan_gia = st.button("Dự đoán giá")
    if du_doan_gia:
        # In ra các thông tin đã chọn
        st.write("Tiêu đề nhà:", tieu_de)
        st.write("Mô tả nhà:", mo_ta)
        st.write("Địa chỉ nhà:", dia_chi)
        st.write("Giá nhà:", gia)
        st.write("Diện tích:", dien_tich)
        st.write("Số phòng ngủ:", so_phong_ngu)
        st.write("Số phòng tắm:", so_phong_tam)
        st.write("Số tầng:", so_tang)

        # Giả sử giá dự đoán là 25000000 VND, thực tế cần dùng mô hình ML để dự đoán
        gia_du_doan = 25000000
        st.write("Giá dự đoán (giả sử), thực tế cần dùng mô hình ML để dự đoán:", gia_du_doan)
    # Làm tiếp cho phần phát hiện nhà bất thường
    st.write("### 2. Phát hiện nhà bất thường")    
    gia_du_doan = st.number_input("Nhập giá dự đoán (VND) để kiểm tra bất thường", min_value=0, max_value=100000000000, value=2500000000, step=10000000)
    kiem_tra_bat_thuong = st.button("Kiểm tra bất thường")
    if kiem_tra_bat_thuong:
        # In ra các thông tin đã chọn    
        st.write("Giá dự đoán (VND):", gia_du_doan)
        # Giả sử nếu số km đã đi > 150000 hoặc giá dự đoán < 5000000 thì là bất thường
        if gia_du_doan < 2000000000 or gia_du_doan > 3000000000:
            st.write("#### Nhà bất thường")
        else:
            st.write("#### Nhà bình thường.")
        # Trên thực tế cần dùng mô hình phát hiện bất thường để kiểm tra
        # Nếu có mô hình ML, có thể gọi hàm dự đoán ở đây
        pass

elif choice=='Gợi ý điều khiển project 2':
    st.write("##### Gợi ý điều khiển project 2: Recommender System")
    st.write("##### Dữ liệu mẫu")
    # Đọc dữ liệu từ file house_samples.csv, chỉ lấy 3 cột id, tieu_de, mo_ta
    df = pd.read_csv("house_samples.csv")    
    
    st.dataframe(df)
    st.write("### 1. Tìm kiếm nhà tương tự")
    # Tạo điều khiển để người dùng chọn nhà
    selected_house = st.selectbox("Chọn nhà", df['tieu_de'])
    st.write("Nhà đã chọn:", selected_house) 
    # Từ nhà đã chọn này, người dùng có thể xem thông tin chi tiết của nhà
    # hoặc thực hiện các xử lý khác
    # tạo điều khiển để người dùng tìm kiếm nhà dựa trên thông tin người dùng nhập
    search = st.text_input("Nhập thông tin tìm kiếm")
    # Tìm kiếm nhà dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    # Trên thực tế sử dụng content-based filtering (cosine similarity/ gensim/hydrid) để tìm kiếm nhà tương tự
    result = df[df['tieu_de'].str.lower().str.contains(search.lower())]    
    # tạo button submit
    tim_kiem = st.button("Tìm kiếm")
    if tim_kiem:
        st.write("Danh sách nhà tìm được:")
        st.dataframe(result)
       
# Done
    
    
    
        

        
        

    



