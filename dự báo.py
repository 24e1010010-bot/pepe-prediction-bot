import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# --- BƯỚC 1: KẾT NỐI & LẤY DỮ LIỆU LỊCH SỬ (>= 2 NĂM) ---
SHEET_ID = "1eNxCsEEQsh7NEpjuaxdHLsNKM8TBKw-RA4h5FmeMe7Q"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def run_prediction_pipeline():
    try:
        df = pd.read_csv(CSV_URL)
        df.columns = [str(c).strip() for c in df.columns]
        print(f"✅ Bước 1: Đã kết nối thành công. Tổng số dòng thô: {len(df)}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    # --- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ---
    # Xác định cột Ngày và Giá (Dựa trên cấu trúc file của bạn)
    date_col = 'Ngành' if 'Ngành' in df.columns else df.columns[0]
    price_col = next((c for c in df.columns if "Giá Đóng" in c or "Close" in c), None)
    
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    
    # Làm sạch số: đổi dấu phẩy sang dấu chấm (chuẩn VN -> quốc tế)
    def clean_price(val):
        s = str(val).replace(',', '.')
        return pd.to_numeric(s, errors='coerce')

    df['Close'] = df[price_col].apply(clean_price)
    
    # Loại bỏ "rác": Giá PEPE thật luôn < 0.1 USD (loại bỏ cột Năm bị nhầm)
    df = df[df['Close'] < 0.1].dropna(subset=[date_col, 'Close']).sort_values(by=date_col)
    print(f"✅ Bước 2: Tiền xử lý xong. Dữ liệu sạch: {len(df)} dòng.")

    # --- BƯỚC 3: TẠO ĐẶC TRƯNG (LAG / INDICATORS) ---
    # 1. Đặc trưng trễ (Lag 1 ngày)
    df['lag_1'] = df['Close'].shift(1)
    # 2. Chỉ báo SMA (Trung bình động 7 ngày)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    # 3. Chỉ báo RSI (Sức mạnh tương đối 14 ngày)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df = df.dropna().reset_index(drop=True)
    print("✅ Bước 3: Đã tạo đặc trưng Lag, SMA và RSI.")

    # --- BƯỚC 4: CHIA TRAIN - TEST THEO THỜI GIAN ---
    # Tỷ lệ: 80% dữ liệu cũ để học, 20% dữ liệu mới nhất để kiểm tra
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    feature_cols = ['Close', 'lag_1', 'SMA_7', 'RSI']
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    def create_xy(data, lookback=15):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    lookback = 15
    X_train, y_train = create_xy(train_scaled, lookback)
    X_test, y_test = create_xy(test_scaled, lookback)
    print("✅ Bước 4: Đã chia dữ liệu Train/Test theo thời gian.")

    # --- BƯỚC 5: HUẤN LUYỆN MÔ HÌNH (LSTM) ---
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("🚀 Bước 5: AI đang học từ lịch sử giá...")
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)

    # --- BƯỚC 6: DỰ ĐOÁN ---
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Đưa về giá trị thực tế (Inverse Scaling)
    def invert(scaled_val):
        dummy = np.zeros((len(scaled_val), len(feature_cols)))
        dummy[:, 0] = scaled_val.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred = invert(y_pred_scaled)
    y_actual = invert(y_test.reshape(-1, 1))
    print("✅ Bước 6: Hoàn thành dự đoán trên tập Test.")

    # --- BƯỚC 7: ĐÁNH GIÁ & TRỰC QUAN HÓA ---
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:\n- MAE: {mae:.10f}\n- RMSE: {rmse:.10f}")

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    test_dates = test_df[date_col].values[lookback:]
    
    plt.plot(test_dates, y_actual, label='Giá Thực Tế (Actual)', color='cyan', lw=2)
    plt.plot(test_dates, y_pred, label='AI Dự Đoán (Predicted)', color='yellow', ls='--')
    
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.8f'))
    plt.title('BÁO CÁO DỰ BÁO PEPE - QUY TRÌNH 7 BƯỚC', color='lime', fontsize=15)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.xticks(rotation=30)
    plt.show()

# Thực thi
run_prediction_pipeline()