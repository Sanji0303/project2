# ==================== RECOMMENDATION ====================
elif menu == "Đề xuất bất động sản":
    st.title("Đề xuất bất động sản")
    
    df = models['df_recommend']
    
    # ==================== THỐNG KÊ DỮ LIỆU ====================
    st.subheader("📊 Thống kê dữ liệu")
    quan_stats = df['quan'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    for i, (quan, count) in enumerate(quan_stats.items()):
        if i == 0:
            with col1:
                st.metric(f"📍 {quan.upper()}", f"{count:,} BĐS")
        elif i == 1:
            with col2:
                st.metric(f"📍 {quan.upper()}", f"{count:,} BĐS")
        else:
            with col3:
                st.metric(f"📍 {quan.upper()}", f"{count:,} BĐS")
    
    st.divider()
    
    # ==================== PHẦN 1: TÌM KIẾM THEO NHU CẦU ====================
    st.subheader("🔍 Tìm kiếm bất động sản theo nhu cầu")
    
    with st.expander("📝 Nhập thông tin nhu cầu", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_quan = st.selectbox(
                "📍 Quận mong muốn:",
                options=["Tất cả"] + df['quan'].unique().tolist(),
                index=0
            )
        
        with col2:
            search_price_min = st.number_input(
                "💰 Giá tối thiểu (tỷ):",
                min_value=0.5,
                max_value=100.0,
                value=0.5,
                step=0.5
            )
            search_price_max = st.number_input(
                "💰 Giá tối đa (tỷ):",
                min_value=0.5,
                max_value=100.0,
                value=20.0,
                step=0.5
            )
        
        with col3:
            search_area_min = st.number_input(
                "📐 Diện tích tối thiểu (m²):",
                min_value=10.0,
                max_value=500.0,
                value=30.0,
                step=5.0
            )
            search_area_max = st.number_input(
                "📐 Diện tích tối đa (m²):",
                min_value=10.0,
                max_value=500.0,
                value=100.0,
                step=5.0
            )
        
        search_btn = st.button("🔍 Tìm kiếm", type="primary")
        
        if search_btn:
            # Lọc theo quận
            if search_quan != "Tất cả":
                search_df = df[df['quan'] == search_quan].copy()
            else:
                search_df = df.copy()
            
            # Lọc theo giá
            search_df = search_df[
                (search_df['gia_ban_num'] >= search_price_min * 1e9) &
                (search_df['gia_ban_num'] <= search_price_max * 1e9)
            ]
            
            # Lọc theo diện tích
            search_df = search_df[
                (search_df['dien_tich_num'] >= search_area_min) &
                (search_df['dien_tich_num'] <= search_area_max)
            ]
            
            if len(search_df) > 0:
                st.success(f"✅ Tìm thấy {len(search_df)} bất động sản phù hợp!")
                
                # Hiển thị kết quả tìm kiếm
                with st.expander("📋 Kết quả tìm kiếm", expanded=True):
                    search_display = search_df.head(10).copy()
                    search_display['display'] = search_display.apply(
                        lambda x: f"[{x['quan'].upper()}] {str(x['tieu_de'])[:60]}... - {x['gia_ban']} | {x['dien_tich']}",
                        axis=1
                    )
                    
                    for i, row in search_display.iterrows():
                        st.write(f"**{i+1}. {row['display']}**")
                    
                    if len(search_df) > 10:
                        st.info(f"Hiển thị 10/ {len(search_df)} kết quả. Vui lòng chọn BĐS trong danh sách bên dưới để xem chi tiết và đề xuất.")
            else:
                st.warning("⚠️ Không tìm thấy bất động sản phù hợp với nhu cầu của bạn. Vui lòng điều chỉnh lại thông tin!")
    
    st.divider()
    
    # ==================== PHẦN 2: CHỌN BĐS ĐỂ ĐỀ XUẤT ====================
    st.subheader("🏠 Chọn bất động sản để xem đề xuất")
    
    # Lấy danh sách quận thực tế từ dữ liệu
    available_quan = df['quan'].unique().tolist()
    
    # Nếu chỉ có 1 quận, hiển thị cảnh báo
    if len(available_quan) == 1:
        st.warning(f"⚠️ Dữ liệu hiện tại chỉ có quận {available_quan[0]}. Vui lòng kiểm tra lại file df_recommend.pkl!")
    
    selected_quan = st.selectbox(
        "Chọn quận:",
        options=available_quan,
        index=0
    )
    
    # Lọc theo quận đã chọn
    df_filtered = df[df['quan'] == selected_quan].copy()
    
    st.info(f"📊 Hiển thị {len(df_filtered)} bất động sản tại quận {selected_quan.upper()}")
    
    # Hiển thị danh sách BĐS
    if len(df_filtered) > 0:
        # Hiển thị tất cả BĐS (không giới hạn 100)
        df_display = df_filtered.copy()
        df_display['display'] = df_display.apply(
            lambda x: f"{str(x['tieu_de'])[:50]}... - {x['gia_ban']} | {x['dien_tich']}", 
            axis=1
        )
        
        selected_idx = st.selectbox(
            "Chọn bất động sản:",
            range(len(df_display)),
            format_func=lambda x: df_display.iloc[x]['display']
        )
        
        with st.expander("📋 Xem chi tiết", expanded=True):
            prop = df_display.iloc[selected_idx]
            st.write(f"**Tiêu đề:** {prop['tieu_de']}")
            st.write(f"**Giá:** {prop['gia_ban']} | **Diện tích:** {prop['dien_tich']} | **Quận:** {prop['quan']}")
        
        n_recommend = st.slider("Số lượng đề xuất:", 3, 10, 5)
        rec_type = st.radio("Loại đề xuất:", ["Hybrid", "Content-based"])
        
        if st.button("🔍 Đề xuất", type="primary"):
            # Lấy index gốc trong dataframe đầy đủ
            original_idx = df_display.iloc[selected_idx].name
            
            sim_matrix = models['hybrid_sim'] if rec_type == "Hybrid" else models['cosine_sim']
            
            # Lấy độ tương đồng
            sim_scores = list(enumerate(sim_matrix[original_idx]))
            
            # Chỉ lấy các BĐS trong cùng quận đã chọn
            same_quan_indices = df_filtered.index.tolist()
            sim_scores = [(idx, score) for idx, score in sim_scores 
                         if idx in same_quan_indices and idx != original_idx]
            
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n_recommend]
            
            if len(sim_scores) == 0:
                st.warning("Không tìm thấy bất động sản tương tự trong cùng quận!")
            else:
                st.divider()
                st.subheader(f"🏠 Kết quả đề xuất tại quận {selected_quan.upper()}:")
                
                for i, (idx, score) in enumerate(sim_scores, 1):
                    prop = df.iloc[idx]
                    st.write(f"**{i}. {str(prop['tieu_de'])[:80]}...**")
                    st.write(f"💰 {prop['gia_ban']} | 📐 {prop['dien_tich']} | 📍 {prop['quan']}")
                    st.write(f"🎯 Độ tương đồng: {score:.3f}")
                    st.divider()
    else:
        st.error(f"❌ Không có dữ liệu bất động sản tại quận {selected_quan}!")
