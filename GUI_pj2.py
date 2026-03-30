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
    page_title="Hệ thống Tư vấn & Tìm kiếm Bất động sản",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ĐƯỜNG DẪN FILE ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_BT1 = os.path.join(BASE_DIR, "file_pkl_bt1")
PATH_BT2 = os.path.join(BASE_DIR, "file_pkl_bt2")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load models và xử lý dữ liệu ngầm (không hiển thị thuật ngữ kỹ thuật ra UI)"""
    models = {}
    
    # Load dữ liệu
    models['df_recommend'] = joblib.load(os.path.join(PATH_BT1, "df_recommend.pkl"))
    
    # Tạo features (ẩn quá trình này với người dùng)
    with st.spinner("Đang khởi tạo hệ thống dữ liệu..."):
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        models['features'] = tfidf.fit_transform(models['df_recommend']['tieu_de'].fillna(''))
        models['tfidf_vectorizer'] = tfidf
    
    # Load mô hình phân khúc
    models['scaler'] = joblib.load(os.path.join(PATH_BT2, "scaler_kmeans.pkl"))
    models['kmeans'] = joblib.load(os.path.join(PATH_BT2, "kmeans_model.pkl"))
    models['features_kmeans'] = joblib.load(os.path.join(PATH_BT2, "features_kmeans.pkl"))
    
    # Tạo df_clustered
    df_clustered = models['df_recommend'].copy()
    X_cluster = df_clustered[models['features_kmeans']]
    X_scaled = models['scaler'].transform(X_cluster)
    df_clustered['cluster_kmeans'] = models['kmeans'].predict(X_scaled)
    models['df_clustered'] = df_clustered
    
    # Tạo thông tin phân khúc (ngôn ngữ thương mại)
    cluster_info = {}
    for cluster in sorted(df_clustered['cluster_kmeans'].unique()):
        cluster_data = df_clustered[df_clustered['cluster_kmeans'] == cluster]
        avg_price = cluster_data['gia_ban_num'].mean() / 1e9
        avg_area = cluster_data['dien_tich_num'].mean()
        
        if avg_price < 3:
            segment = "Phổ thông - Nhà nhỏ"
            desc = "Phù hợp đầu tư sinh lời, sinh viên, người độc thân"
            icon = "🏘️"
            price_range = "Dưới 3 tỷ"
            area_range = "Dưới 40m²"
        elif avg_price < 6:
            segment = "Trung cấp - Diện tích vừa"
            desc = "Lựa chọn lý tưởng cho gia đình trẻ, vợ chồng mới cưới"
            icon = "🏠"
            price_range = "3 - 6 tỷ"
            area_range = "40 - 60m²"
        elif avg_price < 10:
            segment = "Khá giả - Không gian rộng"
            desc = "Không gian sống thoải mái cho gia đình 2-3 thế hệ"
            icon = "🏢"
            price_range = "6 - 10 tỷ"
            area_range = "60 - 90m²"
        elif avg_price < 15:
            segment = "Cao cấp - Tiện nghi"
            desc = "Môi trường sống chất lượng cao, an ninh đảm bảo"
            icon = "🏰"
            price_range = "10 - 15 tỷ"
            area_range = "90 - 120m²"
        elif avg_price < 25:
            segment = "Siêu cao cấp - Biệt thự"
            desc = "Khẳng định đẳng cấp, không gian sống sang trọng"
            icon = "🏛️"
            price_range = "15 - 25 tỷ"
            area_range = "120 - 200m²"
        else:
            segment = "Hạng sang - Dinh thự"
            desc = "Bất động sản tinh hoa dành cho giới thượng lưu"
            icon = "👑"
            price_range = "Trên 25 tỷ"
            area_range = "Trên 200m²"
        
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

# Hàm tính độ phù hợp
def get_similar_properties(idx, features, df_filtered, top_k=10):
    query_vector = features[idx]
    similarities = cosine_similarity(query_vector, features).flatten()
    
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    similar_scores = similarities[similar_indices]
    
    results = []
    for i, (sim_idx, score) in enumerate(zip(similar_indices, similar_scores)):
        if sim_idx in df_filtered.index:
            results.append((sim_idx, score))
        if len(results) >= top_k:
            break
    
    return results

# Load models
with st.spinner("Đang kết nối cơ sở dữ liệu..."):
    models = load_models()

# ==================== SIDEBAR MENU & THÔNG TIN ĐỘI NGŨ ====================
st.sidebar.title("🏠 MENU CHÍNH")
menu = st.sidebar.radio(
    "Danh mục chức năng",
    [
        "🌟 Trang chủ", 
        "📊 Tổng quan thị trường", 
        "🎯 Khám phá phân khúc", 
        "🔍 Tìm kiếm & Gợi ý"
    ]
)

# Hiển thị cố định Đội ngũ phát triển ở Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Đội ngũ phát triển")
st.sidebar.info("""
**Dự án: Hệ thống Tư vấn Bất động sản**

👥 **Thành viên:**
- **Đặng Đức Duy** (Dữ liệu)
- **Huỳnh Lê Xuân Ánh** (Hệ thống gợi ý)
- **Nguyễn Thị Tuyết Vân** (Phân tích thị trường)

🎓 **GVHD:** TS. Nguyễn Văn A
📧 **Liên hệ:** contact@realestate.vn
""")
st.sidebar.caption("© 2024 - Real Estate Recommender System")

# ==================== TRANG CHỦ ====================
if menu == "🌟 Trang chủ":
    st.title("🌟 Chào mừng đến với Hệ thống Tư vấn Bất động sản")
    
    st.markdown("""
    Hệ thống của chúng tôi giúp bạn dễ dàng tìm kiếm, định giá và lựa chọn tổ ấm phù hợp nhất tại các khu vực trung tâm TP.HCM. 
    Với công nghệ phân tích dữ liệu thông minh, chúng tôi mang đến cho bạn những gợi ý chính xác và khách quan nhất.
    """)
    
    st.markdown("### 📌 Khu vực hỗ trợ hiện tại")
    st.markdown("- Quận **Bình Thạnh**\n- Quận **Gò Vấp**\n- Quận **Phú Nhuận**")
    
    st.markdown("### 📊 Thống kê kho dữ liệu")
    col1, col2, col3 = st.columns(3)
    df = models['df_recommend']
    with col1:
        st.metric("Tổng số BĐS đang bán", f"{len(df):,} căn")
    with col2:
        st.metric("Mức giá trung bình", f"{df['gia_ban_num'].mean()/1e9:.1f} tỷ VNĐ")
    with col3:
        st.metric("Diện tích trung bình", f"{df['dien_tich_num'].mean():.0f} m²")
        
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True, caption="Giải pháp tìm nhà thông minh cho mọi gia đình")

# ==================== TỔNG QUAN THỊ TRƯỜNG ====================
elif menu == "📊 Tổng quan thị trường":
    st.title("📊 Bức tranh Thị trường Bất động sản")
    
    st.write("Dựa trên hàng ngàn tin đăng hiện có, hệ thống của chúng tôi đã phân tích và chia thị trường thành **6 phân khúc chính** để giúp bạn dễ dàng hình dung và lựa chọn.")
    
    st.subheader("📈 Tỷ trọng các phân khúc")
    cluster_counts = {cluster: info['count'] for cluster, info in models['cluster_info'].items()}
    
    # Chỉ lấy tên phân khúc để hiển thị biểu đồ cho đẹp
    chart_data = pd.DataFrame({
        'Phân khúc': [models['cluster_info'][c]['segment'] for c in sorted(cluster_counts.keys())],
        'Số lượng căn': [cluster_counts[c] for c in sorted(cluster_counts.keys())]
    })
    st.bar_chart(chart_data.set_index('Phân khúc'))
    
    st.subheader("📋 Chi tiết từng phân khúc")
    for cluster in sorted(cluster_counts.keys()):
        info = models['cluster_info'][cluster]
        with st.expander(f"{info['icon']} **{info['segment']}** ({info['count']} căn)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**💰 Khoảng giá:** {info['price_range']}")
                st.write(f"**💵 Giá trung bình:** {info['avg_price']:.1f} tỷ")
            with col2:
                st.write(f"**📐 Diện tích:** {info['area_range']}")
                st.write(f"**📏 DT trung bình:** {info['avg_area']:.0f} m²")
            with col3:
                st.write(f"**🎯 Phù hợp với:**")
                st.write(f"*{info['description']}*")

# ==================== KHÁM PHÁ PHÂN KHÚC ====================
elif menu == "🎯 Khám phá phân khúc":
    st.title("🎯 Định vị Phân khúc Bất động sản")
    
    st.markdown("""
    Bạn đang quan tâm đến một căn nhà nhưng không biết nó thuộc phân khúc nào trên thị trường? 
    Hãy nhập thông tin bên dưới, hệ thống AI của chúng tôi sẽ phân tích và cho bạn câu trả lời.
    """)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        gia = st.number_input("💰 Mức giá dự kiến (tỷ VNĐ)", min_value=0.5, max_value=200.0, value=5.0, step=0.5)
        dien_tich = st.number_input("📐 Diện tích (m²)", min_value=10.0, max_value=1000.0, value=50.0, step=5.0)
    
    with col2:
        quan = st.selectbox("📍 Khu vực (Quận)", ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"])
        st.info(f"💡 Mẹo: Hệ thống sẽ so sánh căn nhà của bạn với hàng ngàn căn khác tại {quan} để đưa ra kết quả chính xác nhất.")
    
    if st.button("🔍 Phân tích ngay", type="primary"):
        gia_num = gia * 1e9
        price_per_m2 = gia_num / dien_tich
        quan_map = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
        quan_encoded = quan_map[quan]
        
        # Xử lý ngầm
        new_data = np.array([[gia_num, dien_tich, price_per_m2, quan_encoded]])
        new_scaled = models['scaler'].transform(new_data)
        cluster_pred = models['kmeans'].predict(new_scaled)[0]
        cluster_info = models['cluster_info'][cluster_pred]
        
        st.divider()
        st.subheader("📊 Kết quả Phân tích")
        
        st.success(f"### {cluster_info['icon']} Bất động sản này thuộc phân khúc: **{cluster_info['segment']}**")
        st.write(f"**💡 Đánh giá:** {cluster_info['description']}")
        
        st.markdown("#### So sánh với mặt bằng chung của phân khúc này:")
        col1, col2 = st.columns(2)
        with col1:
            price_diff = gia - cluster_info['avg_price']
            if price_diff > 0:
                st.warning(f"💰 Giá cao hơn mức trung bình khoảng **{abs(price_diff):.1f} tỷ**")
            elif price_diff < 0:
                st.success(f"💰 Giá tốt hơn mức trung bình khoảng **{abs(price_diff):.1f} tỷ**")
            else:
                st.info("💰 Mức giá sát với trung bình thị trường")
        
        with col2:
            area_diff = dien_tich - cluster_info['avg_area']
            if area_diff > 0:
                st.success(f"📐 Rộng hơn mức trung bình khoảng **{abs(area_diff):.0f} m²**")
            elif area_diff < 0:
                st.warning(f"📐 Nhỏ hơn mức trung bình khoảng **{abs(area_diff):.0f} m²**")
            else:
                st.info("📐 Diện tích đạt chuẩn trung bình")

# ==================== TÌM KIẾM & GỢI Ý ====================
elif menu == "🔍 Tìm kiếm & Gợi ý":
    st.title("🔍 Tìm kiếm & Gợi ý Bất động sản")
    st.write("Hệ thống gợi ý thông minh sẽ giúp bạn tìm được những căn nhà có đặc điểm tương đồng nhất với sở thích của bạn.")
    
    df = models['df_recommend']
    features_matrix = models['features']
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    
    # Form tìm kiếm
    with st.form("search_form"):
        st.subheader("📝 Tiêu chí tìm kiếm của bạn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_quan = st.selectbox("📍 Khu vực", options=["Tất cả"] + df['quan'].unique().tolist())
            price_range = st.selectbox("💰 Tài chính", ["Tất cả", "Dưới 3 tỷ", "3 - 6 tỷ", "6 - 10 tỷ", "10 - 15 tỷ", "15 - 25 tỷ", "Trên 25 tỷ"])
            area_range = st.selectbox("📐 Diện tích mong muốn", ["Tất cả", "Dưới 40 m²", "40 - 60 m²", "60 - 90 m²", "90 - 120 m²", "120 - 200 m²", "Trên 200 m²"])
        
        with col2:
            property_type = st.multiselect("🏢 Loại hình", ["Nhà phố", "Biệt thự", "Căn hộ", "Nhà mặt tiền", "Nhà hẻm"])
            features = st.multiselect("✨ Tiện ích nổi bật", ["Hẻm ô tô", "Mặt tiền", "Nội thất đầy đủ", "Nhà mới xây", "Gần trường học", "Gần chợ"])
            keywords = st.text_input("🔎 Từ khóa tự do", placeholder="Ví dụ: nhà đẹp hẻm ô tô gần chợ")
        
        submitted = st.form_submit_button("🔍 Tìm kiếm nhà phù hợp", type="primary", use_container_width=True)
    
    if submitted:
        with st.spinner("Hệ thống đang quét hàng ngàn tin đăng..."):
            filtered_df = df.copy()
            
            if selected_quan != "Tất cả":
                filtered_df = filtered_df[filtered_df['quan'] == selected_quan]
            
            # Lọc giá
            if price_range != "Tất cả":
                price_map = {"Dưới 3 tỷ": (0, 3e9), "3 - 6 tỷ": (3e9, 6e9), "6 - 10 tỷ": (6e9, 10e9), "10 - 15 tỷ": (10e9, 15e9), "15 - 25 tỷ": (15e9, 25e9), "Trên 25 tỷ": (25e9, float('inf'))}
                min_p, max_p = price_map[price_range]
                filtered_df = filtered_df[(filtered_df['gia_ban_num'] >= min_p) & (filtered_df['gia_ban_num'] <= max_p)]
            
            # Lọc diện tích
            if area_range != "Tất cả":
                area_map = {"Dưới 40 m²": (0, 40), "40 - 60 m²": (40, 60), "60 - 90 m²": (60, 90), "90 - 120 m²": (90, 120), "120 - 200 m²": (120, 200), "Trên 200 m²": (200, float('inf'))}
                min_a, max_a = area_map[area_range]
                filtered_df = filtered_df[(filtered_df['dien_tich_num'] >= min_a) & (filtered_df['dien_tich_num'] <= max_a)]
            
            st.session_state.search_results = filtered_df
            st.session_state.selected_property = None
    
    # Hiển thị kết quả
    if st.session_state.search_results is not None:
        filtered_df = st.session_state.search_results
        
        if len(filtered_df) == 0:
            st.warning("⚠️ Rất tiếc, chưa tìm thấy bất động sản nào khớp hoàn toàn với tiêu chí của bạn. Hãy thử nới lỏng các yêu cầu nhé!")
        else:
            st.success(f"✅ Tuyệt vời! Tìm thấy **{len(filtered_df)}** căn nhà phù hợp với bạn.")
            
            st.divider()
            st.subheader("🏠 Chọn một căn nhà bạn thích để xem các gợi ý tương tự")
            
            filtered_df['display'] = filtered_df.apply(
                lambda x: f"🏠 {str(x['tieu_de'])[:80]}... | Giá: {x['gia_ban']} | DT: {x['dien_tich']} | {x['quan']}",
                axis=1
            )
            
            selected_idx = st.selectbox(
                "Danh sách nhà phù hợp:",
                options=range(len(filtered_df)),
                format_func=lambda x: filtered_df.iloc[x]['display']
            )
            
            selected_prop = filtered_df.iloc[selected_idx]
            
            with st.expander("📋 Thông tin chi tiết căn nhà đang chọn", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**🏷️ Tiêu đề:** {selected_prop['tieu_de']}")
                    st.write(f"**💰 Giá bán:** {selected_prop['gia_ban']}")
                    st.write(f"**📐 Diện tích:** {selected_prop['dien_tich']}")
                with col2:
                    st.write(f"**📍 Vị trí:** {selected_prop['quan']}")
                    if 'cluster_kmeans' in selected_prop.index:
                        cluster = selected_prop['cluster_kmeans']
                        segment_name = models['cluster_info'].get(cluster, {}).get('segment', 'Chưa xác định')
                        st.write(f"**🌟 Phân khúc:** {segment_name}")
            
            st.subheader("🎯 Có thể bạn cũng sẽ thích những căn nhà này")
            n_recommend = st.slider("Số lượng gợi ý bạn muốn xem:", 3, 10, 5)
            
            if st.button("✨ Xem danh sách gợi ý", type="primary"):
                with st.spinner("Hệ thống AI đang tìm kiếm những căn nhà có đặc điểm tương đồng nhất..."):
                    original_idx = filtered_df.index.get_loc(selected_prop.name)
                    similar_results = get_similar_properties(original_idx, features_matrix, filtered_df, n_recommend)
                    
                    if len(similar_results) == 0:
                        st.info("Chưa tìm thấy thêm căn nhà nào tương tự trong danh sách lọc hiện tại.")
                    else:
                        for i, (idx, score) in enumerate(similar_results, 1):
                            prop = df.iloc[idx]
                            
                            # Đổi điểm similarity thành % cho dễ hiểu
                            match_percent = score * 100
                            
                            if 'cluster_kmeans' in prop.index:
                                cluster = prop['cluster_kmeans']
                                segment_name = models['cluster_info'].get(cluster, {}).get('segment', 'Chưa xác định')
                            else:
                                segment_name = "Chưa xác định"
                            
                            with st.expander(f"**{i}. {prop['tieu_de'][:80]}...** (Độ phù hợp: {match_percent:.1f}%)", expanded=(i==1)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**💰 Giá bán:** {prop['gia_ban']}")
                                    st.write(f"**📐 Diện tích:** {prop['dien_tich']}")
                                    st.write(f"**📍 Vị trí:** {prop['quan']}")
                                with col2:
                                    st.write(f"**🌟 Phân khúc:** {segment_name}")
                                    st.progress(float(score)) # Thanh tiến trình thể hiện độ phù hợp
                                
                                if 'mo_ta' in prop.index and pd.notna(prop['mo_ta']):
                                    st.write(f"**📝 Mô tả chi tiết:** {prop['mo_ta'][:200]}...")
