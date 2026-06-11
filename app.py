import streamlit as st  # SỬA LỖI: Đã đổi 'tf' thành 'st' chuẩn hóa toàn hệ thống
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CẤU HÌNH TRANG ĐẦU TIÊN (LỆNH ĐẦU TIÊN BẮT BUỘC)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Rủi Ro & Gian Lận",
    page_icon="🛡️"
)

# ==========================================
# IMPORT & HÀM CACHE DÙNG CHUNG
# ==========================================
@st.cache_data(max_entries=5)
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes để tối ưu cache và tránh lỗi đồng bộ hashable.
    Trả về DataFrame đã xử lý định dạng.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu gốc huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tập dữ liệu huấn luyện (.csv, .xlsx)",
        type=["csv", "xlsx"],
        help="Đính kèm file dataset tập mẫu của bạn để trích xuất các biến đặc trưng (X_1 đến X_14) và biến mục tiêu (default)."
    )
    
    st.divider()
    
    # 2. Lựa chọn thuật toán (Trích xuất từ notebook thử nghiệm 3 mô hình)
    st.subheader("🤖 Thuật toán Học máy")
    model_choice = st.selectbox(
        "Chọn mô hình huấn luyện",
        options=["Random Forest", "Logistic Regression", "Decision Tree"],
        index=0,
        help="Lựa chọn một trong ba thuật toán đã được thử nghiệm cấu hình trong notebook."
    )
    
    # 3. Cấu hình siêu tham số động theo mô hình chọn
    st.subheader("🎛️ Tham số mô hình AI")
    
    random_state = st.number_input(
        "Random State",
        value=42,
        step=1,
        help="Cố định tính ngẫu nhiên khi phân tách dữ liệu và chạy thuật toán để tái hiện kết quả."
    )
    
    if model_choice == "Random Forest":
        n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=200, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        criterion = st.selectbox("Tiêu chuẩn đo lường (criterion)", ["gini", "entropy", "log_loss"], index=0)
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=15, help="Độ sâu giới hạn của mỗi cây quyết định.")
        
    elif model_choice == "Logistic Regression":
        penalty = st.selectbox("Phương pháp chuẩn hóa (penalty)", ["l2", "none"], index=0)
        C_val = st.slider("Hệ số nghịch đảo chuẩn hóa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", value=100, min_value=50, max_value=1000, step=50)
        
    elif model_choice == "Decision Tree":
        criterion = st.selectbox("Tiêu chuẩn đo lường (criterion)", ["gini", "entropy"], index=0)
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=10)
        min_samples_split = st.slider("Mẫu tối thiểu để phân tách", min_value=2, max_value=20, value=2)

    st.divider()
    
    # 4. Nút kích hoạt hành động huấn luyện duy nhất
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True, help="Bấm để kích hoạt luồng Train/Test Split và khớp mô hình dựa trên tham số đã chọn.")

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Hệ Thống Dự Báo Rủi Ro Tín Dụng & Phát Hiện Gian Lận")
st.caption("Ứng dụng phân tích dữ liệu giao dịch đặc trưng, tự động hóa quy trình huấn luyện học máy và đưa ra cảnh báo sớm các tài khoản có nguy cơ vỡ nợ (default) hoặc gian lận.")

if uploaded_file is None:
    st.info("💡 **Hướng dẫn:** Vui lòng tải tập dữ liệu mẫu (Ví dụ: file `dataset1.csv`) tại thanh Sidebar bên trái để bắt đầu kích hoạt ứng dụng.")
    st.stop()
else:
    # Đọc dữ liệu thô qua hàm cache bằng bytes
    file_bytes = uploaded_file.read()
    df = load_data(file_bytes, uploaded_file.name)
    if df is not None:
        st.caption(f"📁 **Đang kết nối tệp dữ liệu nguồn:** `{uploaded_file.name}` | Định dạng hợp lệ.")
    else:
        st.stop()

st.divider()

# XÁC ĐỊNH SCHEMA BIẾN (Tự động trích xuất từ cấu trúc dữ liệu)
features_list = [c for c in df.columns if c.startswith('X_')]
target_col = 'default' if 'default' in df.columns else None

# Kiểm tra tính toàn vẹn của dữ liệu đầu vào theo notebook
if len(features_list) == 0 or target_col is None:
    st.error(f"Cấu trúc file dữ liệu không khớp với thiết kế cấu hình notebook. File cần có các cột biến đặc trưng từ X_1 đến X_14 và biến mục tiêu '{target_col if target_col else 'default'}'.")
    st.stop()

# ==========================================
# KHỐI HUẤN LUYỆN (Chạy khi bấm nút, lưu kết quả vào session_state)
# ==========================================
if train_clicked:
    with st.spinner("🔄 Hệ thống đang phân chia dữ liệu và tiến hành huấn luyện mô hình..."):
        X = df[features_list]
        y = df[target_col]
        
        # Đồng bộ cấu trúc chia Train/Test 75/25 như notebook mẫu ngầm định
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
        
        # Khởi tạo mô hình dựa trên cấu hình người dùng chọn động trên UI
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(penalty=penalty, C=C_val, max_iter=max_iter, random_state=random_state)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
            
        # Khớp mô hình (Fit)
        model.fit(X_train, y_train)
        
        # Chấm điểm dự báo trên tập kiểm thử
        y_pred = model.predict(X_test)
        
        # Lưu trữ trạng thái bền vững vào Streamlit Session State để chia sẻ giữa các Tab độc lập
        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_choice
        st.session_state['features'] = features_list
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['df_source'] = df
        
    st.success(f"🎉 Đã huấn luyện thành công mô hình **{model_choice}**! Các tab phân tích và dự báo phía dưới đã sẵn sàng hiển thị kết quả.")

# ==========================================
# KHỞI TẠO CÁC TAB NỘI DUNG CHÍNH (THÀNH PHẦN 3 -> 6)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến", 
    "🎯 Kết quả kiểm định mô hình", 
    "🔮 Triển khai sử dụng mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab1:
    st.subheader("📋 Thống kê mô tả & Cấu trúc tập mẫu")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col_m1.metric("Số dòng (Hàng mẫu)", f"{df.shape[0]:,}")
    col_m2.metric("Số cột thuộc tính", f"{df.shape[1]}")
    col_m3.metric("Dung lượng File", f"{file_size_mb:.2f} MB")
    
    st.write("📂 **Xem nhanh 5 bản ghi dữ liệu mẫu đầu tiên (Head):**")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.write("📊 **Bảng phân tích thống kê toán học các thuộc tính biến đặc trưng:**")
    selected_cols = features_list + [target_col]
    st.dataframe(df[selected_cols].describe().T, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab2:
    st.subheader("📉 Biểu đồ phân tích phân phối thuộc tính")
    st.write("Phân tích trực quan hóa phân phối lớp rủi ro mục tiêu và đặc trưng dữ liệu giao dịch:")
    
    col_b1, col_b2 = st.columns(2)
    col_b3, col_b4 = st.columns(2)
    
    # Thêm config={'displayModeBar': False} để ẩn thanh công cụ thừa của Plotly
    with col_b1:
        target_counts = df[target_col].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng khách hàng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: '0: An toàn', 1: '1: Rủi ro/Gian lận'})
        fig1 = px.bar(target_counts, x='Trạng thái', y='Số lượng khách hàng', 
                      title="Phân phối biến mục tiêu (default)",
                      color='Trạng thái', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        
    with col_b2:
        fig2 = px.histogram(df, x='X_1', color=target_col, barmode='overlay',
                            title="Phân phối thuộc tính liên tục X_1 theo nhóm Rủi ro",
                            labels={'X_1': 'Chỉ số đặc trưng X_1'},
                            color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        
    with col_b3:
        fig3 = px.box(df, x=target_col, y='X_5', color=target_col,
                      title="Biểu đồ Boxplot phân tích giá trị ngoại lai X_5",
                      labels={'default': 'Nhãn Rủi Ro', 'X_5': 'Giá trị biến X_5'})
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        
    with col_b4:
        fig4 = px.histogram(df, x='X_13', title="Biểu đồ tần suất mật độ của thuộc tính X_13",
                            marginal="rug", color_discrete_sequence=['#4A90E2'])
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

    st.write("---")
    st.markdown("🔍 **Tùy chọn nâng cao: Khảo sát biểu đồ các biến khác**")
    all_features_to_select = st.multiselect("Chọn các biến đặc trưng bạn muốn vẽ biểu đồ mật độ bổ sung:", options=features_list, default=['X_2', 'X_6'])
    if all_features_to_select:
        for idx, var in enumerate(all_features_to_select):
            fig_dynamic = px.histogram(df, x=var, color=target_col, title=f"Tần suất phân phối thuộc tính đặc trưng {var}", barmode='group')
            st.plotly_chart(fig_dynamic, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng thuật toán học máy")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ **Thông báo:** Chưa tìm thấy dữ liệu mô hình huấn luyện. Vui lòng chọn thuật toán và bấm nút **[🚀 Huấn luyện mô hình]** ở thanh Sidebar bên trái để xem kết quả kiểm định chi tiết.")
    else:
        model = st.session_state['trained_model']
        m_name = st.session_state['model_name']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        
        st.success(f"📊 Kết quả phân tích đánh giá chất lượng của mô hình: **{m_name}**")
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        st.write("✨ **Các chỉ số phân loại chính yếu (Classification Metrics):**")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Độ chính xác tổng thể (Accuracy)", f"{accuracy:.2%}")
        col_s2.metric("Độ chính xác nhãn rủi ro (Precision)", f"{precision:.2%}")
        col_s3.metric("Độ nhạy bắt gian lận (Recall)", f"{recall:.2%}")
        col_s4.metric("F1-Score (Nhãn rủi ro)", f"{f1_score:.2f}")
        
        st.write("---")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("🧩 **Ma trận nhầm lẫn (Confusion Matrix):**")
            cm_df = pd.DataFrame(cm, index=['Thực tế An toàn (0)', 'Thực tế Rủi ro (1)'], columns=['Dự đoán An toàn (0)', 'Dự đoán Rủi ro (1)'])
            st.dataframe(cm_df, use_container_width=True)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Dự đoán An toàn (0)', 'Dự đoán Rủi ro (1)'],
                y=['Thực tế An toàn (0)', 'Thực tế Rủi ro (1)'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                showscale=False
            ))
            fig_cm.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})
            
        with col_c2:
            st.write("📄 **Báo cáo phân loại chi tiết (Classification Report):**")
            rep_dict = classification_report(y_test, y_pred, output_dict=True)
            rep_df = pd.DataFrame(rep_dict).transpose()
            st.dataframe(rep_df.style.format(precision=3), use_container_width=True)
            
        if hasattr(model, 'feature_importances_'):
            st.write("---")
            st.write("🌿 **Độ quan trọng đóng góp của các thuộc tính đầu vào (Feature Importances):**")
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Thuộc tính biến': features_list,
                'Độ quan trọng (Tỷ trọng)': importances
            }).sort_values(by='Độ quan trọng (Tỷ trọng)', ascending=True)
            
            fig_imp = px.bar(feat_imp_df, x='Độ quan trọng (Tỷ trọng)', y='Thuộc tính biến', orientation='h',
                             title="Xếp hạng mức độ tác động của các biến X đến quyết định phân loại rủi ro",
                             color='Độ quan trọng (Tỷ trọng)', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab4:
    st.subheader("🔮 Chấm điểm dữ liệu mới & Đưa ra dự báo")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ **Thông báo:** Vui lòng thực hiện thao tác huấn luyện mô hình ở Sidebar trước khi sử dụng tính năng dự báo rủi ro thực tế.")
    else:
        model = st.session_state['trained_model']
        features_list = st.session_state['features']
        
        mode = st.radio("Chọn chế độ nạp dữ liệu cần dự đoán:", 
                        ["Chế độ 1: Nhập trực tiếp thông số 1 khách hàng", 
                         "Chế độ 2: Tải file danh sách khách hàng hàng loạt"],
                        horizontal=True)
        
        st.write("---")
        
        if mode == "Chế độ 1: Nhập trực tiếp thông số 1 khách hàng":
            st.markdown("📝 **Nhập các chỉ số thông tin thuộc tính của hồ sơ giao dịch:**")
            source_df = st.session_state['df_source']
            
            with st.form("single_prediction_form"):
                col_inputs = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(features_list):
                    col_idx = idx % 3
                    min_v = float(source_df[feat].min())
                    max_v = float(source_df[feat].max())
                    mean_v = float(source_df[feat].median())
                    
                    with col_inputs[col_idx]:
                        input_data[feat] = st.number_input(
                            f"Chỉ số {feat}",
                            min_value=min_v,
                            max_value=max_v,
                            value=mean_v,
                            format="%.4f",
                            help=f"Khoảng giá trị thực tế trong tập mẫu: [{min_v:.2f} đến {max_v:.2f}]"
                        )
                
                submit_predict = st.form_submit_button("🔮 Thực hiện chấm điểm dự báo", type="primary", use_container_width=True)
                
            if submit_predict:
                single_df = pd.DataFrame([input_data])
                pred_label = model.predict(single_df)[0]
                
                st.write("📊 **Kết quả phân tích hồ sơ:**")
                if pred_label == 1:
                    st.error("🚨 **CẢNH BÁO RỦI RO CAO:** Tài khoản giao dịch này được mô hình phân loại thuộc nhóm có hành vi **Nguy cơ cao / Có biến động bất thường hoặc Vỡ nợ** (default = 1).")
                else:
                    st.success("✅ **HỒ SƠ AN TOÀN:** Hệ thống đánh giá khách hàng thuộc nhóm lành tính, **Không có nguy cơ vỡ nợ hoặc gian lận** (default = 0).")
                    
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(single_df)[0]
                    st.info(f"📈 **Xác suất phân tích chi tiết từ mô hình AI:** Tỷ lệ An toàn: {prob[0]:.2%} | Tỷ lệ Rủi ro/Gian lận: {prob[1]:.2%}")

        # CHẾ ĐỘ 2: DỰ BÁO HÀNG LOẠT QUA FILE EXCEL/CSV (Đã bọc try-except an toàn)
        else:
            st.markdown("📁 **Tải lên tệp dữ liệu tập khách hàng mới cần quét rủi ro số lượng lớn:**")
            st.caption("Lưu ý: Cấu trúc file tải lên phải chứa đầy đủ các cột thuộc tính đặc trưng từ `X_1` đến `X_14` tương tự file mẫu.")
            
            batch_file = st.file_uploader("Tải tệp danh sách cần chấm điểm rủi ro hàng loạt", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                # SỬA LỖI/CẢI TIẾN: Thêm xử lý bọc lỗi khi đọc file danh sách lạ
                try:
                    if batch_file.name.endswith('.csv'):
                        batch_df = pd.read_csv(batch_file)
                    else:
                        batch_df = pd.read_excel(batch_file)
                except Exception as err:
                    st.error(f"❌ Không thể phân tích cấu trúc file này. Vui lòng kiểm tra lại định dạng tệp! Chi tiết lỗi: {err}")
                    st.stop()
                
                missing_cols = [col for col in features_list if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"❌ File tải lên không đúng cấu trúc định dạng biến! Thiếu các cột bắt buộc sau: {missing_cols}")
                else:
                    st.success("✅ Kiểm tra cấu trúc file trùng khớp hoàn toàn. Tiến hành chấm điểm phân loại tự động...")
                    
                    X_batch = batch_df[features_list]
                    predictions = model.predict(X_batch)
                    
                    result_df = batch_df.copy()
                    result_df['DỰ_BÁO_MÔ_HÌNH'] = predictions
                    result_df['KẾT_LUẬN_HỆ_THỐNG'] = result_df['DỰ_BÁO_MÔ_HÌNH'].map({0: 'An toàn', 1: 'Cảnh báo Nguy cơ Rủi ro'})
                    
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_batch)
                        result_df['XÁC_SUẤT_RỦI_RO_GAI_LẬN'] = probabilities[:, 1]
                    
                    total_records = result_df.shape[0]
                    risk_records = np.sum(predictions == 1)
                    risk_ratio = risk_records / total_records
                    
                    col_b_m1, col_b_m2, col_b_m3 = st.columns(3)
                    col_b_m1.metric("Tổng số hồ sơ đã quét", f"{total_records:,} tài khoản")
                    col_b_m2.metric("Số hồ sơ phát hiện nguy cơ", f"{risk_records:,} hồ sơ", delta=f"{risk_records} ca phát hiện", delta_color="inverse")
                    col_b_m3.metric("Tỷ lệ rủi ro phát hiện", f"{risk_ratio:.2%}")
                    
                    st.write("📊 **Bảng chi tiết kết quả phân loại từ hệ thống AI (Xem nhanh):**")
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                    
                    st.download_button(
                        label="📥 Tải xuống bảng kết quả dự báo toàn diện (.CSV)",
                        data=csv_bytes,
                        file_name="ket_qua_du_bao_rui_ro_giao_dich.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
