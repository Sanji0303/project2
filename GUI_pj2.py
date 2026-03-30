# GUI_pj2.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """Load models - sử dụng tính similarity on-the-fly để tránh file lớn"""
    models = {}
    
    # Load BT1 - chỉ load df_recommend
    models['df_recommend'] = joblib.load(os.path.join(PATH_BT1, "df_recommend.pkl"))
    
    # Tạo TF-IDF features từ tiêu đề
    with st.spinner("Đang xây dựng features từ dữ liệu..."):
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        models['features'] = tfidf.fit_transform(models['df_recommend']['tieu_de'].fillna(''))
        models['tfidf_vectorizer'] = tfidf
    
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

# Hàm tính similarity on-the-fly
def get_similar_properties(idx, features, df_filtered, top_k=10):
    """Tính similarity trực tiếp từ feature vectors"""
    query_vector = features[idx]
    similarities = cosine_similarity(query_vector, features).flatten()
    
    # Lấy top_k indices (bỏ qua chính nó)
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    similar_scores = similarities[similar_indices]
    
    # Lọc chỉ giữ các BĐS trong df_filtered
    results = []
    for i, (sim_idx, score) in enumerate(zip(similar_indices, similar_scores)):
        if sim_idx in df_filtered.index:
            results.append((sim_idx, score))
        if len(results) >= top_k:
            break
    
    return results

# Load models
with st.spinner("Đang tải mô hình và xây dựng features..."):
    models = load_models()

# ==================== SIDEBAR MENU ====================
st.sidebar.title("🏠 MENU")
menu = st.sidebar.radio(
    "Chọn chức năng",
    ["📊 Bài toán kinh doanh", "📈 Đánh giá Mô hình", "🔮 Dự đoán phân cụm", "🏠 Đề xuất bất động sản", "👥 Đội ngũ phát triển"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Đội ngũ phát triển")
st.sidebar.markdown("""
- **Đặng Đức Duy** - Xử lý dữ liệu
- **Huỳnh Lê Xuân Ánh** - Hệ thống đề xuất
- **Nguyễn Thị Tuyết Vân** - Phân cụm BĐS
""")
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 - Hệ thống Đề xuất & Phân cụm Bất động sản")

# ==================== BÀI TOÁN KINH DOANH ====================
if menu == "📊 Bài toán kinh doanh":
    st.title("📊 Bài toán Kinh doanh")
    
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
    
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# ==================== ĐÁNH GIÁ MÔ HÌNH ====================
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
        
        st.subheader("📊 Phân bố số lượng theo 6 cụm")
        cluster_counts = {cluster: info['count'] for cluster, info in models['cluster_info'].items()}
        chart_data = pd.DataFrame({
            'Cụm': [f"Cụm {c}" for c in sorted(cluster_counts.keys())],
            'Số lượng': [cluster_counts[c] for c in sorted(cluster_counts.keys())]
        })
        st.bar_chart(chart_data.set_index('Cụm'))
        
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
        - Đã tối ưu dung lượng (tính similarity on-the-fly)
        """)
    
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# ==================== DỰ ĐOÁN PHÂN CỤM ====================
elif menu == "🔮 Dự đoán phân cụm":
    st.title("🔮 Dự đoán phân cụm - Xác định phân khúc")
    
    with st.expander("📌 Hướng dẫn sử dụng & 6 phân khúc BĐS", expanded=True):
        st.markdown("""
        **Cách sử dụng:**
        1. Nhập giá bán và diện tích
        2. Chọn quận
        3. Nhấn "Dự đoán" để xem BĐS thuộc cụm nào
        
        **🏘️ 6 Phân khúc Bất động sản:**
        """)
        
        cluster_info = models['cluster_info']
        col1, col2 = st.columns(2)
        clusters_sorted = sorted(cluster_info.keys())
        mid = len(clusters_sorted) // 2
        
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        gia = st.number_input("💰 Giá (tỷ)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
        dien_tich = st.number_input("📐 Diện tích (m²)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
    
    with col2:
        quan = st.selectbox("📍 Quận", ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"])
        st.info(f"💡 Giá tham khảo {quan}: 5 - 25 tỷ")
    
    if st.button("🔮 Dự đoán cụm", type="primary"):
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        quan_map = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
        quan_encoded = quan_map[quan]
        
        features = models['features_kmeans']
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        new_scaled = models['scaler'].transform(new_data)
        cluster_pred = models['kmeans'].predict(new_scaled)[0]
        cluster_info = models['cluster_info'][cluster_pred]
        
        st.divider()
        st.subheader("📊 Kết quả dự đoán")
        
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
        
        st.divider()
        st.subheader("📊 So sánh với đặc điểm cụm")
        
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
        
        st.divider()
        st.subheader("💡 Gợi ý")
        st.info(f"BĐS này phù hợp với khách hàng trong phân khúc {cluster_info['segment']}. Hãy tham khảo các BĐS tương tự trong mục 'Đề xuất bất động sản'.")
    
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# ==================== ĐỀ XUẤT BẤT ĐỘNG SẢN ====================
elif menu == "🏠 Đề xuất bất động sản":
    st.title("🏠 Tìm kiếm & Đề xuất bất động sản")
    
    df = models['df_recommend']
    features_matrix = models['features']
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'selected_quan' not in st.session_state:
        st.session_state.selected_quan = "Tất cả"
    if 'keywords_input' not in st.session_state:
        st.session_state.keywords_input = ""
    
    # Form tìm kiếm
    with st.form("search_form"):
        st.subheader("🔍 Nhập thông tin nhu cầu của bạn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quan_options = ["Tất cả"] + df['quan'].unique().tolist()
            selected_quan = st.selectbox("📍 Quận", options=quan_options, index=0)
            
            price_range = st.selectbox(
                "💰 Khoảng giá",
                options=[
                    "Tất cả", "Dưới 3 tỷ", "3 - 6 tỷ", "6 - 10 tỷ",
                    "10 - 15 tỷ", "15 - 25 tỷ", "Trên 25 tỷ"
                ],
                index=0
            )
            
            area_range = st.selectbox(
                "📐 Diện tích",
                options=[
                    "Tất cả", "Dưới 40 m²", "40 - 60 m²", "60 - 90 m²",
                    "90 - 120 m²", "120 - 200 m²", "Trên 200 m²"
                ],
                index=0
            )
        
        with col2:
            property_type = st.multiselect(
                "🏢 Loại hình",
                options=[
                    "Nhà phố", "Biệt thự", "Căn hộ", "Nhà mặt tiền",
                    "Nhà hẻm", "Nhà ngõ", "Nhà riêng", "Penthouse"
                ]
            )
            
            features = st.multiselect(
                "✨ Tiện ích & Đặc điểm",
                options=[
                    "Hẻm ô tô", "Mặt tiền", "Nội thất đầy đủ", "Chưa có nội thất",
                    "Thiết kế hiện đại", "Nhà mới xây", "Gần trường học",
                    "Gần chợ", "Gần bệnh viện", "Khu dân cư an ninh"
                ]
            )
            
            keywords = st.text_input(
                "🔎 Từ khóa tìm kiếm",
                placeholder="Ví dụ: nhà đẹp hẻm ô tô gần chợ",
                value=st.session_state.keywords_input
            )
            
            if keywords:
                keyword_count = len([k.strip() for k in keywords.split() if k.strip()])
                st.caption(f"📝 Đã nhập {keyword_count} từ khóa")
        
        submitted = st.form_submit_button("🔍 Tìm kiếm & Đề xuất", type="primary", use_container_width=True)
    
    if submitted:
        with st.spinner("Đang tìm kiếm bất động sản phù hợp..."):
            filtered_df = df.copy()
            st.session_state.keywords_input = keywords
            
            if selected_quan != "Tất cả":
                filtered_df = filtered_df[filtered_df['quan'] == selected_quan]
            
            # Lọc giá
            if price_range != "Tất cả":
                price_map = {
                    "Dưới 3 tỷ": (None, 3e9),
                    "3 - 6 tỷ": (3e9, 6e9),
                    "6 - 10 tỷ": (6e9, 10e9),
                    "10 - 15 tỷ": (10e9, 15e9),
                    "15 - 25 tỷ": (15e9, 25e9),
                    "Trên 25 tỷ": (25e9, None)
                }
                min_price, max_price = price_map[price_range]
                if min_price:
                    filtered_df = filtered_df[filtered_df['gia_ban_num'] >= min_price]
                if max_price:
                    filtered_df = filtered_df[filtered_df['gia_ban_num'] <= max_price]
            
            # Lọc diện tích
            if area_range != "Tất cả":
                area_map = {
                    "Dưới 40 m²": (None, 40),
                    "40 - 60 m²": (40, 60),
                    "60 - 90 m²": (60, 90),
                    "90 - 120 m²": (90, 120),
                    "120 - 200 m²": (120, 200),
                    "Trên 200 m²": (200, None)
                }
                min_area, max_area = area_map[area_range]
                if min_area:
                    filtered_df = filtered_df[filtered_df['dien_tich_num'] >= min_area]
                if max_area:
                    filtered_df = filtered_df[filtered_df['dien_tich_num'] <= max_area]
            
            # Lọc loại hình
            if property_type:
                type_pattern = '|'.join(property_type)
                mask = filtered_df['tieu_de'].str.contains(type_pattern, case=False, na=False)
                if 'mo_ta' in filtered_df.columns:
                    mask = mask | filtered_df['mo_ta'].str.contains(type_pattern, case=False, na=False)
                filtered_df = filtered_df[mask]
            
            # Lọc từ khóa
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
            
            # Lọc tiện ích
            if features:
                feature_pattern = '|'.join(features)
                if 'mo_ta' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['mo_ta'].str.contains(feature_pattern, case=False, na=False)]
            
            st.session_state.search_results = filtered_df
            st.session_state.selected_quan = selected_quan
            st.session_state.selected_property = None
    
    # Hiển thị kết quả
    if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
        filtered_df = st.session_state.search_results
        
        st.success(f"✅ Tìm thấy {len(filtered_df)} bất động sản phù hợp!")
        
        if st.session_state.keywords_input:
            keyword_list = [k.strip() for k in st.session_state.keywords_input.split() if k.strip()]
            if keyword_list:
                st.info(f"🔍 Đã tìm BĐS chứa từ khóa: **{', '.join(keyword_list)}**")
        
        if 'cluster_kmeans' in filtered_df.columns:
            st.subheader("📊 Phân bố theo phân khúc")
            cluster_dist = filtered_df['cluster_kmeans'].value_counts().sort_index()
            cols = st.columns(min(len(cluster_dist), 6))
            for i, (cluster, count) in enumerate(cluster_dist.items()):
                if i < 6:
                    info = models['cluster_info'].get(cluster, {})
                    with cols[i]:
                        st.metric(f"Cụm {cluster}", f"{count} BĐS", help=info.get('segment', f'Cụm {cluster}'))
        
        st.divider()
        st.subheader("🏠 Chọn bất động sản để xem đề xuất tương tự")
        
        filtered_df['display'] = filtered_df.apply(
            lambda x: f"🏠 {str(x['tieu_de'])[:80]}... | {x['gia_ban']} | {x['dien_tich']} | {x['quan']}",
            axis=1
        )
        
        current_index = 0
        if st.session_state.selected_property is not None:
            try:
                current_index = filtered_df.index.get_loc(st.session_state.selected_property)
            except:
                current_index = 0
        
        selected_idx = st.selectbox(
            "Danh sách bất động sản phù hợp:",
            options=range(len(filtered_df)),
            format_func=lambda x: filtered_df.iloc[x]['display'],
            index=current_index,
            key="property_selector"
        )
        
        selected_prop = filtered_df.iloc[selected_idx]
        st.session_state.selected_property = selected_prop.name
        
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
        
        st.subheader("🎯 Đề xuất bất động sản tương tự")
        n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5, key="recommend_slider")
        
        if st.button("🔍 Đề xuất ngay", type="primary", key="recommend_button"):
            with st.spinner("Đang tìm kiếm bất động sản tương tự..."):
                original_idx = filtered_df.index.get_loc(selected_prop.name)
                
                similar_results = get_similar_properties(original_idx, features_matrix, filtered_df, n_recommend)
                
                if len(similar_results) == 0:
                    st.warning("Không tìm thấy bất động sản tương tự!")
                else:
                    st.success(f"✨ Tìm thấy {len(similar_results)} bất động sản tương tự:")
                    
                    for i, (idx, score) in enumerate(similar_results, 1):
                        prop = df.iloc[idx]
                        
                        if 'cluster_kmeans' in prop.index:
                            cluster = prop['cluster_kmeans']
                            cluster_info_model = models['cluster_info'].get(cluster, {})
                            cluster_badge = f"`{cluster_info_model.get('segment', f'Cụm {cluster}')}`"
                        else:
                            cluster_badge = "`Chưa phân cụm`"
                        
                        with st.expander(f"**{i}. {prop['tieu_de'][:80]}...**", expanded=(i==1)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**💰 Giá:** {prop['gia_ban']}")
                                st.write(f"**📐 Diện tích:** {prop['dien_tich']}")
                                st.write(f"**📍 Quận:** {prop['quan']}")
                            with col2:
                                st.write(f"**🏷️ Phân khúc:** {cluster_badge}")
                                st.write(f"**🎯 Độ tương đồng:** `{score:.3f}`")
                            
                            if 'mo_ta' in prop.index and pd.notna(prop['mo_ta']):
                                st.write(f"**📝 Mô tả:** {prop['mo_ta'][:150]}...")
    
    elif st.session_state.search_results is not None and len(st.session_state.search_results) == 0:
        st.warning("⚠️ Không tìm thấy bất động sản phù hợp với nhu cầu của bạn.")
        
        with st.expander("💡 Gợi ý cải thiện tìm kiếm"):
            st.markdown("""
            **Có thể bạn nên:**
            - **Mở rộng khoảng giá** - Thử tăng/giảm khoảng giá tìm kiếm
            - **Mở rộng diện tích** - Điều chỉnh diện tích phù hợp hơn
            - **Giảm bớt tiêu chí** - Bỏ bớt một số loại hình hoặc tiện ích
            - **Đơn giản hóa từ khóa** - Thử dùng từ khóa ngắn gọn hơn
            - **Chọn tất cả quận** - Mở rộng khu vực tìm kiếm
            """)
    
    st.markdown("---")
    st.caption("👥 **Đội ngũ phát triển:** Đặng Đức Duy | Huỳnh Lê Xuân Ánh | Nguyễn Thị Tuyết Vân")

# ==================== ĐỘI NGŨ PHÁT TRIỂN ====================
elif menu == "👥 Đội ngũ phát triển":
    st.title("👥 Đội ngũ phát triển")
    
    st.markdown("""
    ### 🎓 Giảng viên hướng dẫn
    **TS. Nguyễn Văn A** - Trưởng bộ môn Khoa học Dữ liệu
    
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
