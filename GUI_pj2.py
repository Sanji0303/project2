# ==================== TÌM KIẾM & GỢI Ý ====================
elif menu == "🔍 Tìm kiếm & Gợi ý":
    st.title("🔍 Tìm kiếm & Gợi ý Bất động sản")
    st.write("Hệ thống gợi ý thông minh sẽ giúp bạn tìm được những căn nhà có đặc điểm tương đồng nhất với sở thích của bạn.")
    
    # Tạo 2 tab: Nhập thủ công và Upload file
    tab1, tab2 = st.tabs(["📝 Nhập thủ công", "📂 Upload file CSV (Gợi ý hàng loạt)"])
    
    # ==================== TAB 1: NHẬP THỦ CÔNG ====================
    with tab1:
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
                
                # Lọc từ khóa (nếu có)
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
                
                # Lọc loại hình
                if property_type:
                    type_pattern = '|'.join(property_type)
                    mask = filtered_df['tieu_de'].str.contains(type_pattern, case=False, na=False)
                    if 'mo_ta' in filtered_df.columns:
                        mask = mask | filtered_df['mo_ta'].str.contains(type_pattern, case=False, na=False)
                    filtered_df = filtered_df[mask]
                
                # Lọc tiện ích
                if features:
                    feature_pattern = '|'.join(features)
                    if 'mo_ta' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['mo_ta'].str.contains(feature_pattern, case=False, na=False)]
                
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
                                        st.progress(float(score))
                                    
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
                    """Làm sạch dữ liệu batch cho gợi ý"""
                    
                    # Chuẩn hóa tên cột
                    rename_map = {
                        'tiêu đề': 'tieu_de', 'tiêu đề': 'tieu_de', 'title': 'tieu_de',
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
                                    "Số lượng gợi ý": 0,
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
                            
                            # Lưu kết quả
                            result_row = {
                                "STT": idx + 1,
                                "Tiêu đề BĐS cần tìm": row['tieu_de'][:80],
                                "Giá (tỷ)": row['gia_ban'] if row['gia_ban'] > 0 else "Không có",
                                "Diện tích (m²)": row['dien_tich'] if row['dien_tich'] > 0 else "Không có",
                                "Quận": row['quan'],
                            }
                            
                            for i, (pos, score) in enumerate(zip(top_indices, top_scores), 1):
                                if score > 0.01:  # Chỉ hiển thị nếu độ tương đồng > 1%
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
