<<<<<<< HEAD
# app_final.py
=======
# app.py - Phiên bản chỉ dùng Hybrid, bỏ cosine_sim hoàn toàn
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="Hệ thống Đề xuất & Phân cụm Bất động sản", 
    layout="wide"
)

# ==================== ĐƯỜNG DẪN FILE ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_BT1 = os.path.join(BASE_DIR, "file_pkl_bt1")
PATH_BT2 = os.path.join(BASE_DIR, "file_pkl_bt2")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load models với 6 cụm"""
    models = {}
    
    # Load BT1
    models['df_recommend'] = joblib.load(os.path.join(PATH_BT1, "df_recommend.pkl"))
    models['hybrid_sim'] = joblib.load(os.path.join(PATH_BT1, "hybrid_sim.pkl"))
    
    # Load BT2 - 3 file clustering
    models['scaler'] = joblib.load(os.path.join(PATH_BT2, "scaler_kmeans.pkl"))
    models['kmeans'] = joblib.load(os.path.join(PATH_BT2, "kmeans_model.pkl"))
    models['features_kmeans'] = joblib.load(os.path.join(PATH_BT2, "features_kmeans.pkl"))
    
    # Tạo df_clustered
    df_clustered = models['df_recommend'].copy()
    X_cluster = df_clustered[models['features_kmeans']]
    X_scaled = models['scaler'].transform(X_cluster)
    df_clustered['cluster_kmeans'] = models['kmeans'].predict(X_scaled)
    models['df_clustered'] = df_clustered
    
    # Tạo cluster info cho 6 cụm
    cluster_info = {}
    for cluster in sorted(df_clustered['cluster_kmeans'].unique()):
        cluster_data = df_clustered[df_clustered['cluster_kmeans'] == cluster]
        avg_price = cluster_data['gia_ban_num'].mean() / 1e9
        avg_area = cluster_data['dien_tich_num'].mean()
        
        # Phân loại phân khúc dựa trên giá
        if avg_price < 3:
            segment = "🏠 Giá rẻ - Nhà nhỏ"
            desc = "Phù hợp đầu tư, sinh viên, người độc thân"
            icon = "🏘️"
            price_range = "< 3 tỷ"
            area_range = "< 40m²"
        elif avg_price < 6:
            segment = "🏡 Trung cấp - Diện tích vừa"
            desc = "Phù hợp gia đình trẻ, vợ chồng mới cưới"
            icon = "🏠"
            price_range = "3 - 6 tỷ"
            area_range = "40 - 60m²"
        elif avg_price < 10:
            segment = "🏢 Khá giả - Diện tích rộng"
            desc = "Phù hợp gia đình có 2-3 con"
            icon = "🏢"
            price_range = "6 - 10 tỷ"
            area_range = "60 - 90m²"
        elif avg_price < 15:
            segment = "🏰 Cao cấp - Không gian sống tốt"
            desc = "Phù hợp gia đình lớn, khách hàng tài chính tốt"
            icon = "🏰"
            price_range = "10 - 15 tỷ"
            area_range = "90 - 120m²"
        elif avg_price < 25:
            segment = "🏛️ Siêu cao cấp - Biệt thự"
            desc = "Phù hợp giám đốc, doanh nhân, gia đình đa thế hệ"
            icon = "🏛️"
            price_range = "15 - 25 tỷ"
            area_range = "120 - 200m²"
        else:
            segment = "👑 Hạng sang - Dinh thự"
            desc = "Phù hợp đại gia, nhà đầu tư BĐS cao cấp"
            icon = "👑"
            price_range = "> 25 tỷ"
            area_range = "> 200m²"
        
        cluster_info[cluster] = {
            'segment': segment,
            'description': desc,
            'icon': icon,
            'avg_price': avg_price,
            'avg_area': avg_area,
            'count': len(cluster_data),
            'price_range': price_range,
            'area_range': area_range
        }
    
    models['cluster_info'] = cluster_info
    
    return models

# Load models
with st.spinner("Đang tải mô hình..."):
    models = load_models()

# ==================== SIDEBAR MENU ====================
st.sidebar.title("🏠 MENU")
menu = st.sidebar.radio(
<<<<<<< HEAD
    "Chọn chức năng",
    ["📊 Bài toán kinh doanh", "📈 Đánh giá Mô hình", "🔮 Dự đoán phân cụm", "🏠 Đề xuất bất động sản", "👥 Đội ngũ phát triển"]
)

# Thêm thông tin nhóm ở sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Đội ngũ phát triển")
st.sidebar.markdown("""
- **Đặng Đức Duy** - Xử lý dữ liệu
- **Huỳnh Lê Xuân Ánh** - Hệ thống đề xuất
- **Nguyễn Thị Tuyết Vân** - Phân cụm BĐS
""")
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 - Hệ thống Đề xuất & Phân cụm Bất động sản")

# ==================== CÁC MENU ====================

# 1. BÀI TOÁN KINH DOANH
if menu == "📊 Bài toán kinh doanh":
    st.title("📊 Bài toán Kinh doanh")
=======
    "MENU",
    ["🏢 Bài toán kinh doanh", "📊 Đánh giá Mô hình", "🎯 Dự đoán phân cụm", "🔍 Đề xuất bất động sản", "👥 Info Team"]
)

# ==================== BUSINESS PROBLEM ====================
if menu == "🏢 Bài toán kinh doanh":
    st.title("🏢 Bài toán Kinh doanh")
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
    
    st.markdown(f"""
    ### 📌 Vấn đề
    - Khách hàng có nhu cầu mua nhà tại các quận **Bình Thạnh, Gò Vấp, Phú Nhuận**
    - Cần hệ thống đề xuất nhà phù hợp
    - Cần phân cụm chi tiết để hiểu rõ 6 phân khúc thị trường
    
    ### 🎯 Mục tiêu
    1. Phân cụm BĐS thành 6 nhóm chi tiết
    2. Đề xuất nhà dựa trên nội dung và đặc điểm
    3. Hybrid Recommender (nội dung + giá + vị trí)
    
    ### 📊 Dữ liệu
    - **{len(models['df_recommend']):,}** bất động sản tại 3 quận
    - Thuộc tính: giá, diện tích, số phòng, quận
    - Mô tả chi tiết từ tin đăng
    """)
    
    col1, col2, col3 = st.columns(3)
    df = models['df_recommend']
    with col1:
        st.metric("Tổng số BĐS", f"{len(df):,}")
    with col2:
        st.metric("Giá trung bình", f"{df['gia_ban_num'].mean()/1e9:.1f} tỷ")
    with col3:
        st.metric("Diện tích TB", f"{df['dien_tich_num'].mean():.0f} m²")
<<<<<<< HEAD
=======

# ==================== EVALUATION ====================
elif menu == "📊 Đánh giá Mô hình":
    st.title("📊 Đánh giá Mô hình")
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
    
    # Footer thông tin nhóm
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# 2. ĐÁNH GIÁ MÔ HÌNH
elif menu == "📈 Đánh giá Mô hình":
    st.title("📈 Đánh giá Mô hình")
    
    tab1, tab2 = st.tabs(["Clustering (6 Cụm)", "Recommendation (Hybrid)"])
    
    with tab1:
        st.subheader("Đánh giá phân cụm KMeans với 6 cụm")
        
        st.metric("Silhouette Score", "0.4796")
        
        st.info("""
        **📌 Giải thích Silhouette Score:**
        - **> 0.5**: Cấu trúc cụm tốt ✅
        - **0.3 - 0.5**: Cấu trúc cụm trung bình
        - **< 0.3**: Cấu trúc cụm yếu
        
        **KMeans với 6 cụm đạt 0.4796 → Cấu trúc cụm ở mức trung bình khá!**
        """)
        
<<<<<<< HEAD
        # Hiển thị phân bố số lượng
        st.subheader("📊 Phân bố số lượng theo 6 cụm")
        cluster_counts = {cluster: info['count'] for cluster, info in models['cluster_info'].items()}
        
        # Tạo bar chart
        chart_data = pd.DataFrame({
            'Cụm': [f"Cụm {c}" for c in sorted(cluster_counts.keys())],
            'Số lượng': [cluster_counts[c] for c in sorted(cluster_counts.keys())]
        })
        st.bar_chart(chart_data.set_index('Cụm'))
        
        # Hiển thị chi tiết từng cụm
        st.subheader("📈 Chi tiết 6 phân khúc")
        
        for cluster in sorted(cluster_counts.keys()):
            info = models['cluster_info'][cluster]
            with st.expander(f"{info['icon']} **Cụm {cluster}: {info['segment']}**"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Số lượng:** {info['count']} BĐS")
                    st.write(f"**Giá TB:** {info['avg_price']:.1f} tỷ")
                with col2:
                    st.write(f"**Diện tích TB:** {info['avg_area']:.0f} m²")
                    st.write(f"**Khoảng giá:** {info['price_range']}")
                with col3:
                    st.write(f"**Khoảng diện tích:** {info['area_range']}")
                    st.write(f"**🎯 {info['description']}**")
=======
        with col3:
            st.metric("Agglomerative", f"{info['agg_score']:.4f}")
            st.write("✅ **Tốt nhất**")
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
    
    with tab2:
        st.subheader("Đánh giá Recommender")
        st.write("""
        **Hybrid Recommender:** Kết hợp 3 yếu tố
        - Nội dung (TF-IDF): 50%
        - Giá: 25%
        - Vị trí: 25%
        
        **✅ Ưu điểm:**
        - Đề xuất đa chiều, không chỉ dựa trên nội dung
        - Phù hợp với thị trường bất động sản
        - Đã tối ưu dung lượng (bỏ cosine_sim.pkl)
        """)
<<<<<<< HEAD
    
    # Footer thông tin nhóm
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# 3. DỰ ĐOÁN PHÂN CỤM
elif menu == "🔮 Dự đoán phân cụm":
    st.title("🔮 Dự đoán phân cụm - Xác định phân khúc")
    
    with st.expander("📌 Hướng dẫn sử dụng & 6 phân khúc BĐS", expanded=True):
        st.markdown("""
=======

# ==================== PREDICTION ====================
elif menu == "🎯 Dự đoán phân cụm":
    st.title("🎯 Dự đoán phân cụm")
    
    # Thêm thông tin hướng dẫn
    with st.expander("📌 Hướng dẫn", expanded=False):
        st.write("""
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
        **Cách sử dụng:**
        1. Nhập giá bán và diện tích
        2. Chọn quận
        3. Nhấn "Dự đoán" để xem BĐS thuộc cụm nào
        
        **🏘️ 6 Phân khúc Bất động sản:**
        """)
        
        # Hiển thị 6 cụm trong hướng dẫn
        cluster_info = models['cluster_info']
        
        # Hiển thị dạng grid 2 cột
        col1, col2 = st.columns(2)
        
        clusters_sorted = sorted(cluster_info.keys())
        mid = len(clusters_sorted) // 2
        
        # Cột 1
        with col1:
            for cluster in clusters_sorted[:mid]:
                info = cluster_info[cluster]
                st.markdown(f"""
                **{info['icon']} Cụm {cluster}: {info['segment']}**
                - 💰 Giá: {info['price_range']}
                - 📐 Diện tích: {info['area_range']}
                - 🎯 {info['description']}
                ---
                """)
        
        # Cột 2
        with col2:
            for cluster in clusters_sorted[mid:]:
                info = cluster_info[cluster]
                st.markdown(f"""
                **{info['icon']} Cụm {cluster}: {info['segment']}**
                - 💰 Giá: {info['price_range']}
                - 📐 Diện tích: {info['area_range']}
                - 🎯 {info['description']}
                ---
                """)
    
    # Form nhập liệu
    col1, col2 = st.columns(2)
    
    with col1:
        gia = st.number_input("💰 Giá (tỷ)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
        dien_tich = st.number_input("📐 Diện tích (m²)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
    
    with col2:
        quan = st.selectbox("📍 Quận", ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"])
<<<<<<< HEAD
        st.info(f"💡 Giá tham khảo {quan}: 5 - 25 tỷ")
    
    if st.button("🔮 Dự đoán cụm", type="primary"):
        # Tính toán features
=======
        # Thêm thông tin giá tham khảo theo quận
        gia_tham_khao = {
            "Bình Thạnh": "6.5 - 20 tỷ",
            "Gò Vấp": "5.5 - 18 tỷ",
            "Phú Nhuận": "6 - 22 tỷ"
        }
        st.info(f"💡 Giá tham khảo quận {quan.replace('_', ' ').title()}: {gia_tham_khao[quan]}")
    
    if st.button("🔮 Dự đoán", type="primary"):
        # Tính toán
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        
        quan_map = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
        quan_encoded = quan_map[quan]
        
        # Tạo feature vector
        features = models['features_kmeans']
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        
<<<<<<< HEAD
        # Scale và dự đoán
        new_scaled = models['scaler'].transform(new_data)
        cluster_pred = models['kmeans'].predict(new_scaled)[0]
        
        # Lấy thông tin cụm
        cluster_info = models['cluster_info'][cluster_pred]
=======
        # Dự đoán
        kmeans_pred = models['kmeans'].predict(new_scaled)[0]
        gmm_pred = models['gmm'].predict(new_scaled)[0]
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
        
        st.divider()
        st.subheader("📊 Kết quả dự đoán")
        
<<<<<<< HEAD
        # Hiển thị kết quả nổi bật
        st.success(f"### {cluster_info['icon']} BĐS thuộc **Cụm {cluster_pred}**")
        st.info(f"### {cluster_info['segment']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Giá TB cụm", f"{cluster_info['avg_price']:.1f} tỷ")
        with col2:
            st.metric("Diện tích TB cụm", f"{cluster_info['avg_area']:.0f} m²")
        with col3:
            st.metric("Số lượng BĐS", f"{cluster_info['count']:,}")
        
        st.write(f"**📈 Phân tích:** {cluster_info['description']}")
        
        # So sánh với BĐS nhập
=======
        # Hiển thị kết quả chi tiết
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 🎯 KMeans")
            st.metric("Phân cụm", f"Cụm {kmeans_pred}")
            
            if kmeans_pred == 0:
                st.success("🏠 **Phân khúc phổ thông**")
                st.write("""
                **Đặc điểm:**
                - Giá: ~6.5 tỷ
                - Diện tích: ~48 m²
                - Phù hợp: Gia đình trẻ, đầu tư
                """)
            else:
                st.success("🏰 **Phân khúc cao cấp**")
                st.write("""
                **Đặc điểm:**
                - Giá: ~20 tỷ
                - Diện tích: ~114 m²
                - Phù hợp: Gia đình lớn, cao cấp
                """)
        
        with col_b:
            st.markdown("### 📊 GMM")
            st.metric("Phân cụm", f"Cụm {gmm_pred}")
            st.warning("⚠️ Độ tin cậy thấp hơn KMeans")
            st.write("**Silhouette score:** 0.369")
        
        # Thêm thông tin so sánh
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
        st.divider()
        st.subheader("📊 So sánh với đặc điểm cụm")
        
<<<<<<< HEAD
        col1, col2 = st.columns(2)
        with col1:
            price_diff = gia - cluster_info['avg_price']
            if price_diff > 0:
                st.warning(f"💰 Giá nhập cao hơn giá TB cụm {abs(price_diff):.1f} tỷ")
            elif price_diff < 0:
                st.success(f"💰 Giá nhập thấp hơn giá TB cụm {abs(price_diff):.1f} tỷ")
            else:
                st.info("💰 Giá nhập bằng giá TB cụm")
        
        with col2:
            area_diff = dien_tich - cluster_info['avg_area']
            if area_diff > 0:
                st.warning(f"📐 Diện tích lớn hơn TB cụm {abs(area_diff):.0f} m²")
            elif area_diff < 0:
                st.success(f"📐 Diện tích nhỏ hơn TB cụm {abs(area_diff):.0f} m²")
            else:
                st.info("📐 Diện tích bằng TB cụm")
        
        # Gợi ý
        st.divider()
        st.subheader("💡 Gợi ý")
        st.info(f"BĐS này phù hợp với khách hàng trong phân khúc {cluster_info['segment']}. Hãy tham khảo các BĐS tương tự trong mục 'Đề xuất bất động sản'.")
    
    # Footer thông tin nhóm
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# ==================== ĐỀ XUẤT BẤT ĐỘNG SẢN ====================
elif menu == "🏠 Đề xuất bất động sản":
    st.title("🏠 Tìm kiếm & Đề xuất bất động sản")
=======
        # So sánh với giá trung bình
        avg_price = models['df_recommend']['gia_ban_num'].mean() / 1e9
        if gia > avg_price:
            st.info(f"💰 Giá nhập ({gia} tỷ) cao hơn giá trung bình thị trường ({avg_price:.1f} tỷ)")
        else:
            st.info(f"💰 Giá nhập ({gia} tỷ) thấp hơn giá trung bình thị trường ({avg_price:.1f} tỷ)")
        
        # Gợi ý dựa trên kết quả
        if kmeans_pred == 0:
            st.success("💡 **Gợi ý:** Đây là phân khúc nhà phổ thông, phù hợp với nhu cầu ở hoặc đầu tư cho thuê.")
        else:
            st.success("💡 **Gợi ý:** Đây là phân khúc nhà cao cấp, phù hợp với khách hàng có tài chính mạnh, tìm kiếm không gian sống rộng rãi.")

# ==================== RECOMMENDATION ====================
elif menu == "🔍 Đề xuất bất động sản":
    st.title("🔍 Đề xuất bất động sản")
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
    
    # Chọn bất động sản
    df = models['df_recommend']
<<<<<<< HEAD
    
    # Khởi tạo session state để lưu kết quả tìm kiếm
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    
    # ==================== FORM TÌM KIẾM ====================
    with st.form("search_form"):
        st.subheader("🔍 Nhập thông tin nhu cầu của bạn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quan_options = ["Tất cả"] + df['quan'].unique().tolist()
            selected_quan = st.selectbox("📍 Quận", options=quan_options, index=0)
            
            price_range = st.selectbox(
                "💰 Khoảng giá",
                options=[
                    "Tất cả",
                    "Dưới 3 tỷ",
                    "3 - 6 tỷ",
                    "6 - 10 tỷ",
                    "10 - 15 tỷ",
                    "15 - 25 tỷ",
                    "Trên 25 tỷ"
                ],
                index=0
            )
            
            area_range = st.selectbox(
                "📐 Diện tích",
                options=[
                    "Tất cả",
                    "Dưới 40 m²",
                    "40 - 60 m²",
                    "60 - 90 m²",
                    "90 - 120 m²",
                    "120 - 200 m²",
                    "Trên 200 m²"
                ],
                index=0
            )
        
        with col2:
            property_type = st.multiselect(
                "🏢 Loại hình",
                options=[
                    "Nhà phố", "Biệt thự", "Căn hộ", "Nhà mặt tiền", 
                    "Nhà hẻm", "Nhà ngõ", "Nhà riêng", "Penthouse"
                ],
                help="Chọn loại hình bạn quan tâm (có thể chọn nhiều)"
            )
            
            features = st.multiselect(
                "✨ Tiện ích & Đặc điểm",
                options=[
                    "Hẻm ô tô", "Mặt tiền", "Nội thất đầy đủ", "Chưa có nội thất",
                    "Thiết kế hiện đại", "Nhà mới xây", "Gần trường học", 
                    "Gần chợ", "Gần bệnh viện", "Khu dân cư an ninh"
                ],
                help="Chọn các đặc điểm bạn mong muốn"
            )
            
            keywords = st.text_input(
                "🔎 Từ khóa tìm kiếm",
                placeholder="Ví dụ: nhà đẹp hẻm ô tô gần chợ",
                help="💡 Nhập nhiều từ khóa cách nhau bằng dấu cách. Hệ thống sẽ tìm BĐS chứa ít nhất một trong các từ đó."
            )
            
            if keywords:
                keyword_count = len([k.strip() for k in keywords.split() if k.strip()])
                st.caption(f"📝 Đã nhập {keyword_count} từ khóa")
        
        # Nút submit
        submitted = st.form_submit_button("🔍 Tìm kiếm & Đề xuất", type="primary", use_container_width=True)
    
    # ==================== XỬ LÝ TÌM KIẾM ====================
    if submitted:
        with st.spinner("Đang tìm kiếm bất động sản phù hợp..."):
            filtered_df = df.copy()
            
            # Lọc theo quận
            if selected_quan != "Tất cả":
                filtered_df = filtered_df[filtered_df['quan'] == selected_quan]
            
            # Lọc theo giá
            if price_range != "Tất cả":
                if price_range == "Dưới 3 tỷ":
                    filtered_df = filtered_df[filtered_df['gia_ban_num'] < 3e9]
                elif price_range == "3 - 6 tỷ":
                    filtered_df = filtered_df[(filtered_df['gia_ban_num'] >= 3e9) & (filtered_df['gia_ban_num'] < 6e9)]
                elif price_range == "6 - 10 tỷ":
                    filtered_df = filtered_df[(filtered_df['gia_ban_num'] >= 6e9) & (filtered_df['gia_ban_num'] < 10e9)]
                elif price_range == "10 - 15 tỷ":
                    filtered_df = filtered_df[(filtered_df['gia_ban_num'] >= 10e9) & (filtered_df['gia_ban_num'] < 15e9)]
                elif price_range == "15 - 25 tỷ":
                    filtered_df = filtered_df[(filtered_df['gia_ban_num'] >= 15e9) & (filtered_df['gia_ban_num'] < 25e9)]
                elif price_range == "Trên 25 tỷ":
                    filtered_df = filtered_df[filtered_df['gia_ban_num'] >= 25e9]
            
            # Lọc theo diện tích
            if area_range != "Tất cả":
                if area_range == "Dưới 40 m²":
                    filtered_df = filtered_df[filtered_df['dien_tich_num'] < 40]
                elif area_range == "40 - 60 m²":
                    filtered_df = filtered_df[(filtered_df['dien_tich_num'] >= 40) & (filtered_df['dien_tich_num'] < 60)]
                elif area_range == "60 - 90 m²":
                    filtered_df = filtered_df[(filtered_df['dien_tich_num'] >= 60) & (filtered_df['dien_tich_num'] < 90)]
                elif area_range == "90 - 120 m²":
                    filtered_df = filtered_df[(filtered_df['dien_tich_num'] >= 90) & (filtered_df['dien_tich_num'] < 120)]
                elif area_range == "120 - 200 m²":
                    filtered_df = filtered_df[(filtered_df['dien_tich_num'] >= 120) & (filtered_df['dien_tich_num'] < 200)]
                elif area_range == "Trên 200 m²":
                    filtered_df = filtered_df[filtered_df['dien_tich_num'] >= 200]
            
            # Lọc theo loại hình
            if property_type:
                type_pattern = '|'.join(property_type)
                mask = filtered_df['tieu_de'].str.contains(type_pattern, case=False, na=False)
                if 'mo_ta' in filtered_df.columns:
                    mask = mask | filtered_df['mo_ta'].str.contains(type_pattern, case=False, na=False)
                filtered_df = filtered_df[mask]
            
            # Lọc theo từ khóa (OR logic)
            if keywords:
                keyword_list = [k.strip() for k in keywords.split() if k.strip()]
                if keyword_list:
                    combined_mask = pd.Series(False, index=filtered_df.index)
                    for kw in keyword_list:
                        title_mask = filtered_df['tieu_de'].str.contains(kw, case=False, na=False)
                        desc_mask = pd.Series(False, index=filtered_df.index)
                        if 'mo_ta' in filtered_df.columns:
                            desc_mask = filtered_df['mo_ta'].str.contains(kw, case=False, na=False)
                        combined_mask = combined_mask | title_mask | desc_mask
                    filtered_df = filtered_df[combined_mask]
            
            # Lọc theo tiện ích
            if features:
                feature_pattern = '|'.join(features)
                if 'mo_ta' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['mo_ta'].str.contains(feature_pattern, case=False, na=False)]
            
            # Lưu kết quả vào session state
            st.session_state.search_results = filtered_df
            st.session_state.selected_quan = selected_quan
            
            # Reset selected property khi tìm kiếm mới
            st.session_state.selected_property = None
    
    # ==================== HIỂN THỊ KẾT QUẢ TÌM KIẾM ====================
    if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
        filtered_df = st.session_state.search_results
        
        st.success(f"✅ Tìm thấy {len(filtered_df)} bất động sản phù hợp!")
        
        # Hiển thị thông tin tìm kiếm
        if 'keywords' in locals() and keywords:
            keyword_list = [k.strip() for k in keywords.split() if k.strip()]
            if keyword_list:
                st.info(f"🔍 Đã tìm BĐS chứa từ khóa: **{', '.join(keyword_list)}**")
        
        # Hiển thị phân bố theo cụm
        if 'cluster_kmeans' in filtered_df.columns:
            st.subheader("📊 Phân bố theo phân khúc")
            cluster_dist = filtered_df['cluster_kmeans'].value_counts().sort_index()
            cols = st.columns(min(len(cluster_dist), 6))
            for i, (cluster, count) in enumerate(cluster_dist.items()):
                if i < 6:
                    info = models['cluster_info'].get(cluster, {})
                    with cols[i]:
                        st.metric(
                            f"Cụm {cluster}", 
                            f"{count} BĐS",
                            help=info.get('segment', f'Cụm {cluster}')
                        )
        
        st.divider()
        
        # ==================== CHỌN BĐS ĐỂ ĐỀ XUẤT ====================
        st.subheader("🏠 Chọn bất động sản để xem đề xuất tương tự")
        
        # Tạo danh sách hiển thị
        filtered_df['display'] = filtered_df.apply(
            lambda x: f"🏠 {str(x['tieu_de'])[:80]}... | {x['gia_ban']} | {x['dien_tich']} | {x['quan']}", 
            axis=1
        )
        
        # Tìm index hiện tại trong session state
        current_index = 0
        if st.session_state.selected_property is not None:
            try:
                current_index = filtered_df.index.get_loc(st.session_state.selected_property)
            except:
                current_index = 0
        
        # Selectbox để chọn BĐS
        selected_idx = st.selectbox(
            "Danh sách bất động sản phù hợp:",
            options=range(len(filtered_df)),
            format_func=lambda x: filtered_df.iloc[x]['display'],
            index=current_index,
            key="property_selector"  # Thêm key để tránh reset
        )
        
        # Lưu selected property vào session state
        selected_prop = filtered_df.iloc[selected_idx]
        st.session_state.selected_property = selected_prop.name
        
        # Hiển thị chi tiết BĐS được chọn
        with st.expander("📋 Xem chi tiết bất động sản đã chọn", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**🏷️ Tiêu đề:** {selected_prop['tieu_de']}")
                st.write(f"**💰 Giá:** {selected_prop['gia_ban']}")
                st.write(f"**📐 Diện tích:** {selected_prop['dien_tich']}")
                st.write(f"**📍 Quận:** {selected_prop['quan']}")
            with col2:
                if 'cluster_kmeans' in selected_prop.index:
                    cluster = selected_prop['cluster_kmeans']
                    cluster_info_model = models['cluster_info'].get(cluster, {})
                    st.write(f"**🏷️ Phân khúc:** {cluster_info_model.get('segment', f'Cụm {cluster}')}")
                if 'price_per_m2' in selected_prop.index:
                    st.write(f"**💵 Giá/m²:** {selected_prop['price_per_m2']/1e6:.1f} triệu/m²")
        
        # ==================== PHẦN ĐỀ XUẤT ====================
        st.subheader("🎯 Đề xuất bất động sản tương tự")
        
        n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5, key="recommend_slider")
        
        if st.button("🔍 Đề xuất ngay", type="primary", key="recommend_button"):
            with st.spinner("Đang tìm kiếm bất động sản tương tự..."):
                original_idx = selected_prop.name
                
                # Lấy similarity matrix
                sim_matrix = models['hybrid_sim']
                sim_scores = list(enumerate(sim_matrix[original_idx]))
                
                # Lọc cùng quận (nếu có chọn quận)
                selected_quan_state = st.session_state.get('selected_quan', "Tất cả")
                if selected_quan_state != "Tất cả":
                    same_quan_indices = filtered_df[filtered_df['quan'] == selected_quan_state].index.tolist()
                else:
                    same_quan_indices = filtered_df.index.tolist()
                
                sim_scores = [(idx, score) for idx, score in sim_scores 
                             if idx in same_quan_indices and idx != original_idx]
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n_recommend]
                
                if len(sim_scores) == 0:
                    st.warning("Không tìm thấy bất động sản tương tự!")
                else:
                    st.success(f"✨ Tìm thấy {len(sim_scores)} bất động sản tương tự:")
                    
                    for i, (idx, score) in enumerate(sim_scores, 1):
                        prop = df.iloc[idx]
                        
                        # Thẻ phân khúc
                        if 'cluster_kmeans' in prop.index:
                            cluster = prop['cluster_kmeans']
                            cluster_info_model = models['cluster_info'].get(cluster, {})
                            cluster_badge = f"`{cluster_info_model.get('segment', f'Cụm {cluster}')}`"
                        else:
                            cluster_badge = "`Chưa phân cụm`"
                        
                        # Tạo expander cho từng đề xuất
                        with st.expander(f"**{i}. {prop['tieu_de'][:80]}...**", expanded=(i==1)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**💰 Giá:** {prop['gia_ban']}")
                                st.write(f"**📐 Diện tích:** {prop['dien_tich']}")
                                st.write(f"**📍 Quận:** {prop['quan']}")
                            with col2:
                                st.write(f"**🏷️ Phân khúc:** {cluster_badge}")
                                st.write(f"**🎯 Độ tương đồng:** `{score:.3f}`")
                            
                            # Hiển thị mô tả nếu có
                            if 'mo_ta' in prop.index and pd.notna(prop['mo_ta']):
                                st.write(f"**📝 Mô tả:** {prop['mo_ta'][:150]}...")
    
    elif st.session_state.search_results is not None and len(st.session_state.search_results) == 0:
        st.warning("⚠️ Không tìm thấy bất động sản phù hợp với nhu cầu của bạn.")
        
        with st.expander("💡 Gợi ý cải thiện tìm kiếm"):
            st.write("""
            **Có thể bạn nên:**
            - **Mở rộng khoảng giá** - Thử tăng/giảm khoảng giá tìm kiếm
            - **Mở rộng diện tích** - Điều chỉnh diện tích phù hợp hơn
            - **Giảm bớt tiêu chí** - Bỏ bớt một số loại hình hoặc tiện ích
            - **Đơn giản hóa từ khóa** - Thử dùng từ khóa ngắn gọn hơn
            - **Chọn tất cả quận** - Mở rộng khu vực tìm kiếm
            """)
    
    # Footer
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")
    
# 5. ĐỘI NGŨ PHÁT TRIỂN
elif menu == "👥 Đội ngũ phát triển":
    st.title("👥 Đội ngũ phát triển")
=======
    df_display = df.head(100).copy()
    df_display['display'] = df_display.apply(
        lambda x: f"{x['tieu_de'][:45]}... - {x['gia_ban']}", axis=1
    )
    
    selected_idx = st.selectbox(
        "Chọn bất động sản:",
        range(len(df_display)),
        format_func=lambda x: df_display.iloc[x]['display']
    )
    
    # Hiển thị chi tiết
    with st.expander("Xem chi tiết", expanded=True):
        prop = df_display.iloc[selected_idx]
        st.write(f"**Tiêu đề:** {prop['tieu_de']}")
        st.write(f"**Giá:** {prop['gia_ban']} | **Diện tích:** {prop['dien_tich']} | **Quận:** {prop['quan']}")
    
    n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5)
    rec_type = st.radio("Loại đề xuất:", ["Hybrid", "Content-based"])
    
    if st.button("Đề xuất", type="primary"):
        sim_matrix = models['hybrid_sim'] if rec_type == "Hybrid" else models['cosine_sim']
        
        sim_scores = list(enumerate(sim_matrix[selected_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommend+1]
        
        st.divider()
        st.subheader("Kết quả đề xuất:")
        
        for i, (idx, score) in enumerate(sim_scores, 1):
            prop = df.iloc[idx]
            st.write(f"**{i}. {prop['tieu_de'][:80]}...**")
            st.write(f"💰 {prop['gia_ban']} | 📐 {prop['dien_tich']} | 📍 {prop['quan']}")
            st.write(f"🎯 Độ tương đồng: {score:.3f}")
            st.divider()

elif menu == "👥 Info Team":
    st.title("👥 Thông tin nhóm")
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
    
    st.markdown("""
    ### 🎓 Giảng viên hướng dẫn
    **TS. Nguyễn Văn A** - Trưởng bộ môn Khoa học Dữ liệu
    
<<<<<<< HEAD
    ### 👨‍💻 Thành viên nhóm
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 👤 Đặng Đức Duy
        - **Vai trò:** Xử lý dữ liệu
        - **Công việc:**
          - Thu thập và làm sạch dữ liệu
          - Xây dựng pipeline xử lý
          - Tối ưu hiệu suất
        """)
    
    with col2:
        st.markdown("""
        #### 👩‍💻 Huỳnh Lê Xuân Ánh
        - **Vai trò:** Hệ thống đề xuất
        - **Công việc:**
          - Xây dựng mô hình TF-IDF
          - Phát triển Hybrid Recommender
          - Đánh giá hiệu quả đề xuất
        """)
    
    with col3:
        st.markdown("""
        #### 👩‍💻 Nguyễn Thị Tuyết Vân
        - **Vai trò:** Phân cụm BĐS
        - **Công việc:**
          - Xây dựng mô hình KMeans, GMM
          - Phân tích đặc điểm từng cụm
          - Trực quan hóa kết quả
        """)
    
    st.divider()
    
    st.markdown("""
    ### 📊 Thông tin dự án
    | Mục | Chi tiết |
    |-----|----------|
    | **Tên dự án** | Hệ thống Đề xuất & Phân cụm Bất động sản |
    | **Công nghệ** | Python, Scikit-learn, Streamlit |
    | **Số lượng BĐS** | 7,881 bất động sản |
    | **Số cụm** | 6 phân khúc |
    | **Silhouette Score** | 0.4796 |
    | **Thời gian thực hiện** | Tháng 10/2024 - 12/2024 |
    
    
    ### 📞 Liên hệ
    - **Email:** project.realestate@gmail.com
    - **GitHub:** github.com/realestate-recommender-system
    """)
    
=======
    **Thành viên:**
    | STT | Họ và tên | Công việc |
    |-----|-----------|-----------|
    | 1 | Đặng Đức Duy | Xử lý dữ liệu  |
    | 2 | [Huỳnh Lê Xuân Ánh ] | Xây dựng models Hệ thống đề xuất |
    | 3 | [Nguyễn Thị Tuyết Vân] | Xây dựng models hệ thống phân cụm BĐS |
                
    **Công nghệ:** 
        Scikit-learn 
            - KMeans 
            - Gaussian Mixture Model (GMM) 
            - Agglomerative Clustering 
        PySpark 
            - KMeans 
            - Gaussian Mixture Model (GMM)
            - BisectingKMeans
    
    """)
    
>>>>>>> fcff86eb06bf6aa781845b2572477610d13a6c05
