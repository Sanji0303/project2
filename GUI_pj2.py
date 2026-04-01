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
segment_counter = {}  # Đếm số lượng cụm trong mỗi phân khúc

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
    segment_counter = {}
    
    for cluster in sorted(df_clustered['cluster_kmeans'].unique()):
        cluster_data = df_clustered[df_clustered['cluster_kmeans'] == cluster]
        avg_price = cluster_data['gia_ban_num'].mean() / 1e9
        avg_area = cluster_data['dien_tich_num'].mean()
        
        # Xác định tên phân khúc cơ bản
        if avg_price < 3:
            base_segment = "Phổ thông - Nhà nhỏ"
            desc = "Phù hợp đầu tư sinh lời, sinh viên, người độc thân"
            icon = "🏘️"
            price_range = "Dưới 3 tỷ"
            area_range = "Dưới 40m²"
        elif avg_price < 6:
            base_segment = "Trung cấp - Diện tích vừa"
            desc = "Lựa chọn lý tưởng cho gia đình trẻ, vợ chồng mới cưới"
            icon = "🏠"
            price_range = "3 - 6 tỷ"
            area_range = "40 - 60m²"
        elif avg_price < 10:
            base_segment = "Khá giả - Không gian rộng"
            desc = "Không gian sống thoải mái cho gia đình 2-3 thế hệ"
            icon = "🏢"
            price_range = "6 - 10 tỷ"
            area_range = "60 - 90m²"
        elif avg_price < 15:
            base_segment = "Cao cấp - Tiện nghi"
            desc = "Môi trường sống chất lượng cao, an ninh đảm bảo"
            icon = "🏰"
            price_range = "10 - 15 tỷ"
            area_range = "90 - 120m²"
        elif avg_price < 25:
            base_segment = "Siêu cao cấp - Biệt thự"
            desc = "Khẳng định đẳng cấp, không gian sống sang trọng"
            icon = "🏛️"
            price_range = "15 - 25 tỷ"
            area_range = "120 - 200m²"
        else:
            base_segment = "Hạng sang - Dinh thự"
            desc = "Bất động sản tinh hoa dành cho giới thượng lưu"
            icon = "👑"
            price_range = "Trên 25 tỷ"
            area_range = "Trên 200m²"
        
        # Đếm số lần xuất hiện của base_segment
        if base_segment not in segment_counter:
            segment_counter[base_segment] = 1
        else:
            segment_counter[base_segment] += 1
        
        # Tạo tên phân biệt nếu có nhiều cụm cùng loại
        if segment_counter[base_segment] > 1:
            segment = f"{base_segment} (Nhóm {segment_counter[base_segment]})"
        else:
            segment = base_segment
        
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
- **Contact: duydd1411@gmail.com**

- **Huỳnh Lê Xuân Ánh** (Hệ thống gợi ý)
- **Contact: huynhlexuananh2002@gmail.com**

- **Nguyễn Thị Tuyết Vân** (Phân tích thị trường)
- **Contact: tuyetvan1418393@gmail.com**
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
    
    # Tạo 2 tab
    tab1, tab2 = st.tabs(["📝 Nhập thủ công", "📂 Upload file CSV (Phân tích hàng loạt)"])
    
    # ==================== TAB 1: NHẬP THỦ CÔNG ====================
    with tab1:
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
    
    # ==================== TAB 2: UPLOAD CSV ====================
    with tab2:
        st.markdown("""
        ### 📂 Phân tích phân khúc hàng loạt bằng file CSV
        
        **Hướng dẫn:**
        1. Upload file CSV của bạn (hệ thống tự động nhận diện cột)
        2. Cần có các cột: **giá** và **diện tích** (có thể đặt tên khác nhau)
        3. Các cột khác (phòng ngủ, số tầng, quận) là tùy chọn
        4. Hệ thống sẽ tự động phân tích và hiển thị biểu đồ
        
        **Hỗ trợ tên cột:** giá, giábán, price, gia_ban | diện tích, dientich, area | phòng ngủ, bedroom | tầng, floor | quận, district
        """)
        
        # Nút tải file mẫu
        if st.button("📥 Tải file mẫu CSV", key="download_template_cluster"):
            sample_data = pd.DataFrame({
                "giá bán (tỷ)": [2.5, 5.8, 8.5, 12.5, 18.0, 35.0, 4.2, 7.5],
                "diện tích (m²)": [35, 55, 72, 85, 110, 250, 45, 68],
                "phòng ngủ": [2, 3, 3, 4, 4, 5, 2, 3],
                "tầng": [1, 2, 2, 3, 3, 4, 1, 2],
                "khu vực": ["Bình Thạnh", "Gò Vấp", "Bình Thạnh", "Phú Nhuận", "Bình Thạnh", "Gò Vấp", "Phú Nhuận", "Bình Thạnh"]
            })
            csv = sample_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải file mẫu (CSV)",
                data=csv,
                file_name="mau_phan_tich_phan_khuc.csv",
                mime="text/csv",
                key="download_cluster_btn"
            )
        
        st.divider()
        
        # Upload file
        uploaded_file = st.file_uploader(
            "📁 Chọn file CSV của bạn",
            type=["csv"],
            help="Hệ thống tự động nhận diện cột. Chỉ cần có cột giá và diện tích là đủ!",
            key="csv_uploader_cluster"
        )
        
        if uploaded_file is not None:
            try:
                # Đọc file gốc
                df_raw = pd.read_csv(uploaded_file)
                st.info(f"📄 File đã tải: {len(df_raw)} dòng, {len(df_raw.columns)} cột")
                
                with st.expander("📋 Xem trước dữ liệu gốc", expanded=False):
                    st.dataframe(df_raw.head(10), use_container_width=True)
                
                # Hàm tự động nhận diện cột
                def auto_detect_columns(df):
                    col_lower = {col: col.lower().strip() for col in df.columns}
                    
                    price_keywords = ['giá', 'gia', 'price', 'giábán', 'gia_ban', 'giá bán', 'giá_trị']
                    price_col = None
                    for col, lower in col_lower.items():
                        if any(kw in lower for kw in price_keywords):
                            price_col = col
                            break
                    
                    area_keywords = ['diện tích', 'dien_tich', 'dientich', 'area', 'dt', 'diện_tích']
                    area_col = None
                    for col, lower in col_lower.items():
                        if any(kw in lower for kw in area_keywords):
                            area_col = col
                            break
                    
                    bedroom_keywords = ['phòng ngủ', 'phong_ngu', 'bedroom', 'pn', 'số_phòng_ngủ']
                    bedroom_col = None
                    for col, lower in col_lower.items():
                        if any(kw in lower for kw in bedroom_keywords):
                            bedroom_col = col
                            break
                    
                    floor_keywords = ['tầng', 'tang', 'floor', 'số_tầng', 'so_tang']
                    floor_col = None
                    for col, lower in col_lower.items():
                        if any(kw in lower for kw in floor_keywords):
                            floor_col = col
                            break
                    
                    district_keywords = ['quận', 'quan', 'district', 'khu_vực', 'kv', 'khu vực']
                    district_col = None
                    for col, lower in col_lower.items():
                        if any(kw in lower for kw in district_keywords):
                            district_col = col
                            break
                    
                    return {
                        'price_col': price_col,
                        'area_col': area_col,
                        'bedroom_col': bedroom_col,
                        'floor_col': floor_col,
                        'district_col': district_col
                    }
                
                # Hàm xử lý giá
                def parse_price(value):
                    if pd.isna(value):
                        return 0
                    if isinstance(value, str):
                        value = value.replace(' tỷ', '').replace('triệu', '').replace(',', '').replace('đ', '').strip()
                    try:
                        num = float(value)
                        if num > 1000:
                            num = num / 1000
                        return num
                    except:
                        return 0
                
                # Hàm xử lý diện tích
                def parse_area(value):
                    if pd.isna(value):
                        return 0
                    if isinstance(value, str):
                        value = value.replace(' m²', '').replace('m2', '').replace(',', '').strip()
                    try:
                        return float(value)
                    except:
                        return 0
                
                # Tự động nhận diện cột
                cols = auto_detect_columns(df_raw)
                
                # Kiểm tra cột bắt buộc
                if cols['price_col'] is None:
                    st.error("❌ Không tìm thấy cột giá! Hãy đảm bảo file có cột chứa thông tin giá (ví dụ: 'giá', 'giá bán', 'price')")
                    st.stop()
                
                if cols['area_col'] is None:
                    st.error("❌ Không tìm thấy cột diện tích! Hãy đảm bảo file có cột chứa thông tin diện tích (ví dụ: 'diện tích', 'area')")
                    st.stop()
                
                # Hiển thị thông tin cột đã nhận diện
                st.success("🔍 **Hệ thống đã nhận diện:**")
                col_info = st.columns(5)
                with col_info[0]:
                    st.write(f"💰 **Giá:** `{cols['price_col']}`")
                with col_info[1]:
                    st.write(f"📐 **Diện tích:** `{cols['area_col']}`")
                with col_info[2]:
                    st.write(f"🛏️ **Phòng ngủ:** `{cols['bedroom_col'] or 'Không có'}`")
                with col_info[3]:
                    st.write(f"🏢 **Số tầng:** `{cols['floor_col'] or 'Không có'}`")
                with col_info[4]:
                    st.write(f"📍 **Quận:** `{cols['district_col'] or 'Không có'}`")
                
                # Xử lý dữ liệu
                with st.spinner("🔄 Đang xử lý và phân tích dữ liệu..."):
                    df_processed = pd.DataFrame()
                    
                    # Xử lý giá
                    df_processed['gia_ban'] = df_raw[cols['price_col']].apply(parse_price)
                    
                    # Xử lý diện tích
                    df_processed['dien_tich'] = df_raw[cols['area_col']].apply(parse_area)
                    
                    # Xử lý số phòng ngủ
                    if cols['bedroom_col']:
                        df_processed['so_phong_ngu'] = pd.to_numeric(df_raw[cols['bedroom_col']], errors='coerce').fillna(2)
                    else:
                        df_processed['so_phong_ngu'] = 2
                    
                    # Xử lý số tầng
                    if cols['floor_col']:
                        df_processed['tong_so_tang'] = pd.to_numeric(df_raw[cols['floor_col']], errors='coerce').fillna(2)
                    else:
                        df_processed['tong_so_tang'] = 2
                    
                    # Xử lý quận
                    if cols['district_col']:
                        df_processed['quan'] = df_raw[cols['district_col']].astype(str).str.strip()
                        quan_map = {'bình thạnh': 'Bình Thạnh', 'gò vấp': 'Gò Vấp', 'phú nhuận': 'Phú Nhuận'}
                        df_processed['quan'] = df_processed['quan'].str.lower().apply(lambda x: quan_map.get(x, 'Gò Vấp'))
                    else:
                        df_processed['quan'] = 'Gò Vấp'
                    
                    # Loại bỏ dòng không hợp lệ
                    invalid_count = len(df_processed[(df_processed['gia_ban'] <= 0) | (df_processed['dien_tich'] <= 0)])
                    df_processed = df_processed[(df_processed['gia_ban'] > 0) & (df_processed['dien_tich'] > 0)]
                    
                    if invalid_count > 0:
                        st.warning(f"⚠️ Đã bỏ qua {invalid_count} dòng có giá trị không hợp lệ")
                    
                    if len(df_processed) == 0:
                        st.error("❌ Không có dữ liệu hợp lệ để phân tích!")
                        st.stop()
                    
                    # Dự đoán phân khúc
                    quan_encode = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
                    clusters = []
                    
                    for _, row in df_processed.iterrows():
                        gia_num = row['gia_ban'] * 1e9
                        price_per_m2 = gia_num / row['dien_tich']
                        quan_encoded = quan_encode.get(row['quan'], 0)
                        
                        new_data = np.array([[gia_num, row['dien_tich'], price_per_m2, quan_encoded]])
                        new_scaled = models['scaler'].transform(new_data)
                        cluster = models['kmeans'].predict(new_scaled)[0]
                        clusters.append(cluster)
                    
                    df_processed['cluster'] = clusters
                    df_processed['phân_khúc'] = df_processed['cluster'].apply(lambda x: models['cluster_info'][x]['segment'])
                    
                    st.success(f"✅ Đã phân tích thành công {len(df_processed)} bất động sản!")
                    
                    with st.expander("📊 Xem trước kết quả phân tích", expanded=False):
                        st.dataframe(df_processed[['gia_ban', 'dien_tich', 'so_phong_ngu', 'tong_so_tang', 'quan', 'phân_khúc']].head(10), use_container_width=True)
                
                # Biểu đồ thống kê
                st.subheader("📊 Thống kê theo phân khúc")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏷️ Phân bố phân khúc**")
                    st.bar_chart(df_processed['phân_khúc'].value_counts())
                
                with col2:
                    st.markdown("**💰 Giá trung bình theo phân khúc**")
                    st.bar_chart(df_processed.groupby('phân_khúc')['gia_ban'].mean().sort_values())
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**📐 Diện tích trung bình theo phân khúc**")
                    st.bar_chart(df_processed.groupby('phân_khúc')['dien_tich'].mean().sort_values())
                
                with col4:
                    st.markdown("**🛏️ Số phòng ngủ trung bình theo phân khúc**")
                    st.bar_chart(df_processed.groupby('phân_khúc')['so_phong_ngu'].mean().sort_values())
                
                col5, col6 = st.columns(2)
                with col5:
                    st.markdown("**🏢 Số tầng trung bình theo phân khúc**")
                    st.bar_chart(df_processed.groupby('phân_khúc')['tong_so_tang'].mean().sort_values())
                
                with col6:
                    st.markdown("**💵 Giá/m² trung bình theo phân khúc**")
                    df_processed['gia_tren_m2'] = df_processed['gia_ban'] / df_processed['dien_tich']
                    st.bar_chart(df_processed.groupby('phân_khúc')['gia_tren_m2'].mean().sort_values())
                
                # Bảng tổng hợp
                st.subheader("📋 Bảng tổng hợp chi tiết")
                
                summary_table = df_processed.groupby('phân_khúc').agg({
                    'gia_ban': ['count', 'mean', 'min', 'max'],
                    'dien_tich': ['mean', 'min', 'max'],
                    'so_phong_ngu': 'mean',
                    'tong_so_tang': 'mean'
                }).round(2)
                
                summary_table.columns = ['Số lượng', 'Giá TB (tỷ)', 'Giá Min (tỷ)', 'Giá Max (tỷ)', 
                                          'Diện tích TB (m²)', 'DT Min (m²)', 'DT Max (m²)',
                                          'Phòng ngủ TB', 'Số tầng TB']
                
                st.dataframe(summary_table, use_container_width=True)
                
                # Tải kết quả
                csv_results = df_processed[['gia_ban', 'dien_tich', 'so_phong_ngu', 'tong_so_tang', 'quan', 'phân_khúc']].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải kết quả phân tích (CSV)",
                    data=csv_results,
                    file_name="ket_qua_phan_tich_phan_khuc.csv",
                    mime="text/csv",
                    key="download_cluster_results"
                )
                
                st.info("💡 **Gợi ý:** Bạn có thể tải kết quả về để phân tích thêm hoặc lưu trữ!")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý file: {str(e)}")
                st.info("Vui lòng kiểm tra lại định dạng file CSV. Hệ thống cần ít nhất 2 cột: giá và diện tích.")


# ==================== TÌM KIẾM & GỢI Ý ====================
elif menu == "🔍 Tìm kiếm & Gợi ý":
    st.title("🔍 Tìm kiếm & Gợi ý Bất động sản")
    st.write("Hệ thống gợi ý thông minh sẽ giúp bạn tìm được những căn nhà có đặc điểm tương đồng nhất với sở thích của bạn.")
    
    # Tạo 2 tab: Nhập thủ công và Upload file
    tab1, tab2 = st.tabs(["📝 Nhập thủ công", "📂 Upload file CSV (Gợi ý hàng loạt)"])
    
    # ==================== TAB 1: NHẬP THỦ CÔNG ====================
    with tab1:
        st.markdown("""
        ### 🔍 Tìm kiếm bất động sản theo nhu cầu
        
        Bạn có thể tìm kiếm theo 2 cách:
        - **Cách 1:** Chọn trực tiếp từ danh sách bất động sản có sẵn
        - **Cách 2:** Nhập các tiêu chí để hệ thống gợi ý những căn phù hợp nhất
        """)
        
        # Tạo 2 sub-tab cho phần nhập thủ công
        sub_tab1, sub_tab2 = st.tabs(["🎯 Chọn từ danh sách có sẵn", "📝 Nhập thông tin chi tiết"])
        
        # ==================== SUB-TAB 1: CHỌN TỪ DANH SÁCH CÓ SẴN ====================
        with sub_tab1:
            st.markdown("##### 🏠 Danh sách bất động sản mới nhất")
            
            df = models['df_recommend']
            features_matrix = models['features']
            
            # Lấy các model cần thiết cho phân khúc
            features_kmeans = models['features_kmeans']
            scaler = models['scaler']
            kmeans = models['kmeans']
            cluster_info = models['cluster_info']
            quan_to_code = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
            
            # Hiển thị danh sách BĐS để chọn
            df_display = df.copy()
            df_display['Hiển thị'] = df_display.apply(
                lambda x: f"🏠 {str(x['tieu_de'])[:70]}... | 💰 {x['gia_ban']} | 📐 {x['dien_tich']} | 📍 {x['quan']}",
                axis=1
            )
            
            selected_prop_name = st.selectbox(
                "📋 Chọn một bất động sản bạn thích:",
                options=df_display.index,
                format_func=lambda x: df_display.loc[x, 'Hiển thị']
            )
            
            selected_prop = df.loc[selected_prop_name]
            
            # Tính phân khúc cho BĐS đã chọn
            selected_features = pd.DataFrame([{
                'gia_ban_num': selected_prop['gia_ban_num'],
                'dien_tich_num': selected_prop['dien_tich_num'],
                'price_per_m2': selected_prop['price_per_m2'],
                'quan_encoded': quan_to_code.get(selected_prop['quan'], 0)
            }])
            selected_scaled = scaler.transform(selected_features[features_kmeans])
            selected_cluster = kmeans.predict(selected_scaled)[0]
            selected_segment = cluster_info[selected_cluster]['segment']
            
            # Hiển thị thông tin chi tiết căn đã chọn
            with st.expander("📋 Xem chi tiết căn nhà đã chọn", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**🏷️ Tiêu đề:** {selected_prop['tieu_de']}")
                    st.write(f"**💰 Giá bán:** {selected_prop['gia_ban']}")
                    st.write(f"**📐 Diện tích:** {selected_prop['dien_tich']}")
                with col2:
                    st.write(f"**📍 Vị trí:** {selected_prop['quan']}")
                    st.write(f"**🌟 Phân khúc:** {selected_segment}")
            
            # Số lượng gợi ý
            n_recommend = st.slider(
                "🔢 Số lượng gợi ý bạn muốn xem:",
                min_value=3, max_value=15, value=5, step=1,
                help="Chọn số lượng bất động sản tương tự sẽ hiển thị"
            )
            
            if st.button("✨ Tìm bất động sản tương tự", type="primary", key="find_similar_btn"):
                with st.spinner("🔍 Hệ thống đang tìm kiếm những căn nhà có đặc điểm tương đồng..."):
                    # Tìm vị trí của BĐS được chọn trong features
                    original_idx = selected_prop_name
                    
                    # Lấy danh sách BĐS cùng quận
                    filtered_df = df[df['quan'] == selected_prop['quan']]
                    
                    # Tìm các BĐS tương tự
                    all_similar = get_similar_properties(original_idx, features_matrix, df, n_recommend + 10)
                    
                    # Lọc chỉ lấy BĐS cùng quận và khác BĐS gốc
                    similar_results = [(idx, score) for idx, score in all_similar 
                                      if idx != original_idx and df.loc[idx, 'quan'] == selected_prop['quan']]
                    
                    # Lấy đúng số lượng yêu cầu
                    similar_results = similar_results[:n_recommend]
                    
                    if len(similar_results) == 0:
                        st.info("ℹ️ Chưa tìm thấy căn nhà nào tương tự trong cùng khu vực.")
                    else:
                        st.success(f"🎉 Tìm thấy {len(similar_results)} căn nhà tương tự!")
                        
                        for i, (idx, score) in enumerate(similar_results, 1):
                            prop = df.loc[idx]
                            match_percent = score * 100
                            
                            # Tính phân khúc cho BĐS được gợi ý
                            prop_features = pd.DataFrame([{
                                'gia_ban_num': prop['gia_ban_num'],
                                'dien_tich_num': prop['dien_tich_num'],
                                'price_per_m2': prop['price_per_m2'],
                                'quan_encoded': quan_to_code.get(prop['quan'], 0)
                            }])
                            prop_scaled = scaler.transform(prop_features[features_kmeans])
                            prop_cluster = kmeans.predict(prop_scaled)[0]
                            prop_segment = cluster_info[prop_cluster]['segment']
                            
                            with st.expander(f"**{i}. {prop['tieu_de'][:80]}...** (Độ phù hợp: {match_percent:.1f}%)", expanded=(i==1)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**💰 Giá bán:** {prop['gia_ban']}")
                                    st.write(f"**📐 Diện tích:** {prop['dien_tich']}")
                                    st.write(f"**📍 Vị trí:** {prop['quan']}")
                                with col2:
                                    st.write(f"**🌟 Phân khúc:** {prop_segment}")
                                    st.progress(min(score, 1.0))
                                
                                if 'mo_ta' in prop.index and pd.notna(prop['mo_ta']):
                                    st.write(f"**📝 Mô tả chi tiết:** {prop['mo_ta'][:200]}...")
        
        # ==================== SUB-TAB 2: NHẬP THÔNG TIN CHI TIẾT ====================
        with sub_tab2:
            st.markdown("##### 📝 Nhập thông tin căn nhà bạn mong muốn")
            
            # Form nhập thông tin
            with st.form("detailed_search_form"):
                st.markdown("**📍 Thông tin cơ bản**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_quan = st.selectbox(
                        "Khu vực", 
                        options=["Tất cả", "Bình Thạnh", "Gò Vấp", "Phú Nhuận"],
                        index=0
                    )
                
                with col2:
                    price_range = st.selectbox(
                        "Tài chính",
                        options=[
                            "Tất cả", "Dưới 3 tỷ", "3 - 6 tỷ", "6 - 10 tỷ",
                            "10 - 15 tỷ", "15 - 25 tỷ", "Trên 25 tỷ"
                        ],
                        index=0
                    )
                
                with col3:
                    area_range = st.selectbox(
                        "Diện tích mong muốn",
                        options=[
                            "Tất cả", "Dưới 40 m²", "40 - 60 m²", "60 - 90 m²",
                            "90 - 120 m²", "120 - 200 m²", "Trên 200 m²"
                        ],
                        index=0
                    )
                
                st.markdown("**🏢 Đặc điểm chi tiết**")
                col4, col5 = st.columns(2)
                
                with col4:
                    property_type = st.multiselect(
                        "Loại hình",
                        options=["Nhà phố", "Biệt thự", "Căn hộ", "Nhà mặt tiền", "Nhà hẻm"],
                        help="Có thể chọn nhiều loại hình"
                    )
                
                with col5:
                    features = st.multiselect(
                        "Tiện ích nổi bật",
                        options=[
                            "Hẻm ô tô", "Mặt tiền", "Nội thất đầy đủ", "Nhà mới xây",
                            "Gần trường học", "Gần chợ", "Gần bệnh viện", "Khu an ninh"
                        ],
                        help="Chọn các tiện ích bạn mong muốn"
                    )
                
                keywords = st.text_input(
                    "🔎 Từ khóa tự do",
                    placeholder="Ví dụ: nhà đẹp, hẻm ô tô, gần chợ, thoáng mát...",
                    help="Nhập các từ khóa cách nhau bằng dấu cách"
                )
                
                st.markdown("---")
                col6, col7, col8 = st.columns([1, 2, 1])
                with col7:
                    n_recommend = st.slider(
                        "🔢 Số lượng gợi ý muốn xem:",
                        min_value=3, max_value=15, value=5, step=1,
                        help="Chọn từ 3 đến 15 căn nhà sẽ được đề xuất"
                    )
                
                submitted = st.form_submit_button("🔍 Tìm kiếm nhà phù hợp", type="primary", use_container_width=True)
            
            if submitted:
                with st.spinner("🔍 Hệ thống đang quét hàng ngàn tin đăng..."):
                    df = models['df_recommend']
                    features_matrix = models['features']
                    filtered_df = df.copy()
                    
                    # 1. Lọc theo quận
                    if selected_quan != "Tất cả":
                        filtered_df = filtered_df[filtered_df['quan'] == selected_quan]
                    
                    # 2. Lọc theo giá
                    if price_range != "Tất cả":
                        price_map = {
                            "Dưới 3 tỷ": (0, 3e9),
                            "3 - 6 tỷ": (3e9, 6e9),
                            "6 - 10 tỷ": (6e9, 10e9),
                            "10 - 15 tỷ": (10e9, 15e9),
                            "15 - 25 tỷ": (15e9, 25e9),
                            "Trên 25 tỷ": (25e9, float('inf'))
                        }
                        min_p, max_p = price_map[price_range]
                        filtered_df = filtered_df[(filtered_df['gia_ban_num'] >= min_p) & (filtered_df['gia_ban_num'] <= max_p)]
                    
                    # 3. Lọc theo diện tích
                    if area_range != "Tất cả":
                        area_map = {
                            "Dưới 40 m²": (0, 40),
                            "40 - 60 m²": (40, 60),
                            "60 - 90 m²": (60, 90),
                            "90 - 120 m²": (90, 120),
                            "120 - 200 m²": (120, 200),
                            "Trên 200 m²": (200, float('inf'))
                        }
                        min_a, max_a = area_map[area_range]
                        filtered_df = filtered_df[(filtered_df['dien_tich_num'] >= min_a) & (filtered_df['dien_tich_num'] <= max_a)]
                    
                    # 4. Lọc theo từ khóa
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
                    
                    # 5. Lọc theo loại hình
                    if property_type:
                        type_pattern = '|'.join(property_type)
                        mask = filtered_df['tieu_de'].str.contains(type_pattern, case=False, na=False)
                        if 'mo_ta' in filtered_df.columns:
                            mask = mask | filtered_df['mo_ta'].str.contains(type_pattern, case=False, na=False)
                        filtered_df = filtered_df[mask]
                    
                    # 6. Lọc theo tiện ích
                    if features:
                        feature_pattern = '|'.join(features)
                        if 'mo_ta' in filtered_df.columns:
                            filtered_df = filtered_df[filtered_df['mo_ta'].str.contains(feature_pattern, case=False, na=False)]
                    
                    # Hiển thị kết quả
                    if len(filtered_df) == 0:
                        st.warning("⚠️ Rất tiếc, không tìm thấy bất động sản nào phù hợp với tiêu chí của bạn. Hãy thử nới lỏng các yêu cầu nhé!")
                    else:
                        st.success(f"✅ Tìm thấy **{len(filtered_df)}** căn nhà phù hợp với tiêu chí của bạn!")
                        
                        # Hiển thị danh sách kết quả tìm kiếm
                        with st.expander("📋 Xem danh sách kết quả tìm kiếm", expanded=False):
                            display_df = filtered_df.copy()
                            display_df['Hiển thị'] = display_df.apply(
                                lambda x: f"🏠 {str(x['tieu_de'])[:70]}... | 💰 {x['gia_ban']} | 📐 {x['dien_tich']} | 📍 {x['quan']}",
                                axis=1
                            )
                            for i, (idx, row) in enumerate(display_df.head(20).iterrows(), 1):
                                st.write(f"{i}. {row['Hiển thị']}")
                            if len(display_df) > 20:
                                st.info(f"... và {len(display_df) - 20} căn khác")
                        
                        st.divider()
                        
                        # ========== PHẦN ĐỀ XUẤT ==========
                        st.subheader(f"🎯 {n_recommend} căn nhà phù hợp nhất với bạn")
                        
                        # Lấy các model cần thiết cho phân khúc
                        features_kmeans = models['features_kmeans']
                        scaler = models['scaler']
                        kmeans = models['kmeans']
                        cluster_info = models['cluster_info']
                        quan_to_code = {"Bình Thạnh": 0, "Gò Vấp": 1, "Phú Nhuận": 2}
                        
                        # Xác định danh sách kết quả đề xuất
                        if keywords:
                            # Tạo vector cho từ khóa tìm kiếm
                            tfidf = models['tfidf_vectorizer']
                            keyword_vector = tfidf.transform([keywords])
                            
                            # Lấy vector của các BĐS trong danh sách lọc
                            filtered_indices = filtered_df.index.tolist()
                            filtered_features = features_matrix[filtered_indices]
                            
                            # Tính similarity với từ khóa
                            similarities = cosine_similarity(keyword_vector, filtered_features).flatten()
                            
                            # Lấy top N
                            top_positions = similarities.argsort()[::-1][:n_recommend]
                            top_scores = similarities[top_positions]
                            top_indices = [filtered_indices[p] for p in top_positions]
                            
                            # Tạo kết quả
                            results = [(idx, score) for idx, score in zip(top_indices, top_scores) if score > 0.05]
                            
                            if len(results) == 0:
                                # Fallback: lấy N căn đầu tiên
                                results = [(filtered_indices[i], 0.5) for i in range(min(n_recommend, len(filtered_indices)))]
                                st.info("💡 Gợi ý dựa trên bộ lọc của bạn (không tìm thấy từ khóa phù hợp)")
                            else:
                                st.info(f"💡 Gợi ý dựa trên từ khóa: **{keywords}**")
                        else:
                            # Nếu không có từ khóa, lấy N căn đầu tiên
                            filtered_indices = filtered_df.index.tolist()
                            results = [(filtered_indices[i], 0.5) for i in range(min(n_recommend, len(filtered_indices)))]
                            st.info("💡 Gợi ý dựa trên bộ lọc của bạn")
                        
                        # Hiển thị kết quả đề xuất
                        if len(results) == 0:
                            st.warning("⚠️ Không có căn nhà nào để hiển thị. Vui lòng thử lại với tiêu chí khác.")
                        else:
                            for i, (idx, score) in enumerate(results, 1):
                                prop = df.loc[idx]
                                match_percent = score * 100
                                
                                # Tính phân khúc cho BĐS được gợi ý
                                prop_features = pd.DataFrame([{
                                    'gia_ban_num': prop['gia_ban_num'],
                                    'dien_tich_num': prop['dien_tich_num'],
                                    'price_per_m2': prop['price_per_m2'],
                                    'quan_encoded': quan_to_code.get(prop['quan'], 0)
                                }])
                                prop_scaled = scaler.transform(prop_features[features_kmeans])
                                prop_cluster = kmeans.predict(prop_scaled)[0]
                                prop_segment = cluster_info[prop_cluster]['segment']
                                
                                with st.expander(f"**{i}. {prop['tieu_de'][:80]}...** (Độ phù hợp: {match_percent:.1f}%)", expanded=(i==1)):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**💰 Giá bán:** {prop['gia_ban']}")
                                        st.write(f"**📐 Diện tích:** {prop['dien_tich']}")
                                        st.write(f"**📍 Vị trí:** {prop['quan']}")
                                    with col2:
                                        st.write(f"**🌟 Phân khúc:** {prop_segment}")
                                        if score < 1.0:
                                            st.progress(min(score, 1.0))
                                    
                                    if 'mo_ta' in prop.index and pd.notna(prop['mo_ta']):
                                        st.write(f"**📝 Mô tả chi tiết:** {prop['mo_ta'][:200]}...")

    
        # ==================== TAB 2: UPLOAD CSV (GỢI Ý HÀNG LOẠT) ====================
    with tab2:
        st.markdown("""
        ### 📂 Gợi ý bất động sản hàng loạt bằng file CSV
        
        **Hướng dẫn:**
        1. Tải file mẫu để tham khảo cấu trúc dữ liệu chuẩn
        2. Hoặc upload file CSV của bạn (hệ thống sẽ tự động nhận diện)
        3. Hệ thống sẽ tìm những căn nhà tương đồng nhất cho từng BĐS trong file
        
        **Các cột hỗ trợ:** tiêu đề, giá bán, diện tích, quận, mô tả (tùy chọn)
        """)
        
        # Nút tải file mẫu
        if st.button("📥 Tải file mẫu CSV", key="download_template_recommend"):
            sample_data = pd.DataFrame({
                "tiêu đề": ["Nhà mặt tiền quận Bình Thạnh", "Căn hộ cao cấp Gò Vấp", "Biệt thự vườn Phú Nhuận"],
                "giá bán (tỷ)": [8.5, 5.2, 25.0],
                "diện tích (m²)": [75, 65, 180],
                "quận": ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"],
                "mô tả": ["Nhà mặt tiền đường lớn", "Gần trường học, chợ", "Biệt thự sân vườn rộng"]
            })
            csv = sample_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải file mẫu (CSV)",
                data=csv,
                file_name="mau_tim_kiem_bds.csv",
                mime="text/csv",
                key="download_sample_btn"
            )
        
        st.divider()
        
        # Upload file
        uploaded_file = st.file_uploader(
            "📁 Chọn file CSV của bạn",
            type=["csv"],
            help="Hệ thống tự động nhận diện cột dữ liệu (tiếng Việt hoặc tiếng Anh).",
            key="csv_uploader_recommend"
        )
        
        if uploaded_file is not None:
            try:
                # Đọc file gốc
                df_raw = pd.read_csv(uploaded_file)
                st.info(f"📄 File đã tải: {len(df_raw)} dòng, {len(df_raw.columns)} cột")
                
                with st.expander("📋 Xem trước dữ liệu gốc", expanded=False):
                    st.dataframe(df_raw.head(10), use_container_width=True)
                
                # ========== HÀM TIỀN XỬ LÝ ==========
                def clean_batch_recommend_data(df):
                    # Chuẩn hóa tên cột
                    rename_map = {
                        'tiêu đề': 'tieu_de', 'title': 'tieu_de',
                        'giá bán': 'gia_ban', 'giá': 'gia_ban', 'price': 'gia_ban',
                        'diện tích': 'dien_tich', 'area': 'dien_tich',
                        'quận': 'quan', 'district': 'quan',
                        'mô tả': 'mo_ta', 'description': 'mo_ta'
                    }
                    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                    
                    # Xử lý cột tiêu đề (bắt buộc)
                    if 'tieu_de' not in df.columns:
                        df['tieu_de'] = "Bất động sản cần tư vấn"
                    
                    # Xử lý cột giá
                    if 'gia_ban' in df.columns:
                        df['gia_ban'] = pd.to_numeric(df['gia_ban'], errors='coerce')
                        df['gia_ban'] = df['gia_ban'].fillna(0)
                    else:
                        df['gia_ban'] = 0
                    
                    # Xử lý cột diện tích
                    if 'dien_tich' in df.columns:
                        df['dien_tich'] = pd.to_numeric(df['dien_tich'], errors='coerce')
                        df['dien_tich'] = df['dien_tich'].fillna(0)
                    else:
                        df['dien_tich'] = 0
                    
                    # Xử lý cột quận
                    if 'quan' not in df.columns:
                        df['quan'] = "Tất cả"
                    
                    # Xử lý cột mô tả
                    if 'mo_ta' not in df.columns:
                        df['mo_ta'] = ""
                    
                    # Điền NaN cho các cột text
                    df['tieu_de'] = df['tieu_de'].fillna("")
                    df['mo_ta'] = df['mo_ta'].fillna("")
                    df['quan'] = df['quan'].fillna("Tất cả")
                    
                    return df
                
                # Tiền xử lý
                with st.spinner("🔄 Đang xử lý dữ liệu..."):
                    df_cleaned = clean_batch_recommend_data(df_raw)
                    
                    with st.expander("📊 Xem trước dữ liệu sau xử lý", expanded=False):
                        st.dataframe(df_cleaned.head(10), use_container_width=True)
                
                # ========== TÌM KIẾM GỢI Ý CHO TỪNG DÒNG ==========
                df_main = models['df_recommend']
                features_matrix = models['features']
                
                # Tạo features cho các BĐS cần tìm
                with st.spinner("🔍 Đang phân tích và tìm kiếm..."):
                    # Kết hợp tiêu đề và mô tả để tìm kiếm
                    search_texts = df_cleaned['tieu_de'] + " " + df_cleaned['mo_ta']
                    
                    # Chuyển thành vector
                    tfidf = models['tfidf_vectorizer']
                    search_features = tfidf.transform(search_texts.fillna(''))
                    
                    results_list = []
                    
                    for idx, row in df_cleaned.iterrows():
                        try:
                            # Lọc theo quận (nếu có)
                            if row['quan'] != "Tất cả" and row['quan'] in df_main['quan'].values:
                                candidates = df_main[df_main['quan'] == row['quan']]
                            else:
                                candidates = df_main
                            
                            if len(candidates) == 0:
                                results_list.append({
                                    "STT": idx + 1,
                                    "Tiêu đề BĐS cần tìm": row['tieu_de'][:50],
                                    "Kết quả": "Không tìm thấy BĐS cùng khu vực"
                                })
                                continue
                            
                            # Lấy vector của BĐS cần tìm
                            query_vector = search_features[idx]
                            
                            # Lấy vector của các BĐS trong danh sách candidates
                            candidate_indices = candidates.index.tolist()
                            candidate_features = features_matrix[candidate_indices]
                            
                            # Tính similarity
                            similarities = cosine_similarity(query_vector, candidate_features).flatten()
                            
                            # Lấy top 5
                            top_indices = similarities.argsort()[::-1][:5]
                            top_scores = similarities[top_indices]
                            
                            # Lưu kết quả - ĐÃ BỎ CỘT GIÁ VÀ DIỆN TÍCH
                            result_row = {
                                "STT": idx + 1,
                                "Tiêu đề BĐS cần tìm": row['tieu_de'][:80],
                                
                            }
                            
                            for i, (pos, score) in enumerate(zip(top_indices, top_scores), 1):
                                if score > 0.01:
                                    prop = candidates.iloc[pos]
                                    result_row[f"Gợi ý {i}"] = f"{prop['tieu_de'][:60]}... | {prop['gia_ban']} | Độ phù hợp: {score*100:.0f}%"
                                else:
                                    result_row[f"Gợi ý {i}"] = "Không đủ độ tương đồng"
                            
                            results_list.append(result_row)
                            
                        except Exception as e:
                            results_list.append({
                                "STT": idx + 1,
                                "Tiêu đề BĐS cần tìm": row['tieu_de'][:50],
                                "Kết quả": f"Lỗi: {str(e)[:50]}"
                            })
                    
                    df_results = pd.DataFrame(results_list)
                
                # Hiển thị kết quả
                st.success(f"✅ Đã tìm kiếm xong {len(df_results)} bất động sản!")
                
                st.subheader("📋 Kết quả gợi ý")
                st.dataframe(df_results, use_container_width=True, height=500)
                
                # Nút tải kết quả
                csv_results = df_results.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải kết quả gợi ý (CSV)",
                    data=csv_results,
                    file_name="ket_qua_goi_y_bds.csv",
                    mime="text/csv",
                    key="download_recommend_results"
                )
                
                st.info("💡 **Mẹo:** Kết quả gợi ý dựa trên độ tương đồng về nội dung mô tả và tiêu đề. Bạn có thể tải về để xem chi tiết!")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý file: {str(e)}")
                st.info("Vui lòng kiểm tra lại định dạng file CSV hoặc tải file mẫu để tham khảo.")
