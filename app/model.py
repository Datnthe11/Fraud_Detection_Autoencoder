import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Đọc dữ liệu
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

data.head(10)
df_no_time= data.drop(columns=['Time'])
# Chuẩn hóa 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_no_time['Amount'] = scaler.fit_transform(df_no_time['Amount'].values.reshape(-1,1))
df_no_time.head()
from sklearn.model_selection import train_test_split

# Chia tập train/test với tỷ lệ 80/20
train_data, test_data = train_test_split(df_no_time, test_size=0.2, random_state=42)

# Chỉ lấy các hàng có class == 0 trong tập train
X_train = train_data[train_data["Class"] == 0].drop(columns=["Class"])

# Tập test vẫn giữ nguyên (có cả class 0 và 1)
X_val,X_test = train_test_split(test_data, test_size=0.5, random_state=42)
X_val = X_val.drop(columns=["Class"])
X_test1 = X_test.drop(columns=["Class"])
y_test1= X_test['Class']
# Kiểm tra kích thước tập dữ liệu
print(f"X_train shape: {X_train.shape}")  # Chỉ chứa class = 0
print(f"X_val shape: {X_val.shape}")    # Có cả class = 0 và 1
print(f"X_test shape: {X_test1.shape}")    # Có cả class = 0 và 1


# Kích thước đầu vào
input_dim = X_train.shape[1]  

# Encoder
input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(256, activation="relu")(input_layer)
encoded = layers.Dense(32, activation="relu")(encoded)
encoded = layers.Dense(16, activation="relu")(encoded)
encoded = layers.Dense(8, activation="relu")(encoded)  # Tầng mã hóa nhỏ hơn

# Decoder
decoded = layers.Dense(16, activation="relu")(encoded)
decoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)  # Output có kích thước bằng đầu vào

# Định nghĩa mô hình Autoencoder
autoencoder = keras.Model(input_layer, decoded)

# Learning rate ban đầu
initial_learning_rate = 0.001  

# Xác định số batch trong mỗi epoch
batch_size = 32
num_samples = X_train.shape[0]
steps_per_epoch = num_samples // batch_size  # Số batch per epoch

# Sử dụng số batch per epoch làm decay_steps
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=steps_per_epoch * 10,  # Giảm LR mỗi 10 epoch
    decay_rate=0.96,  # Mỗi lần giảm còn 96% giá trị trước đó
    staircase=True  # Giảm theo bậc (epoch cụ thể)
)

# Khởi tạo Adam Optimizer với learning rate decay
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile mô hình
autoencoder.compile(optimizer=optimizer, loss="mse")

# Hiển thị kiến trúc mô hình
autoencoder.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Callback Early Stopping: Dừng huấn luyện nếu không cải thiện trong 3 epoch
early_stopping = EarlyStopping(
    monitor="val_loss",  # Theo dõi validation loss
    patience=3,          # Dừng nếu không cải thiện sau 3 epoch
    restore_best_weights=True  # Load lại trọng số tốt nhất
)

# Callback Model Checkpoint: Lưu mô hình tốt nhất
model_checkpoint = ModelCheckpoint(
    "best_autoencoder.keras",  # Tên file lưu
    monitor="val_loss",      # Theo dõi validation loss
    save_best_only=True,     # Chỉ lưu khi tốt hơn phiên bản trước
    mode="min"               # Giảm loss thì tốt hơn
)

# Train mô hình với các callback
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, model_checkpoint]
)

# Tải mô hình đã lưu
model = load_model('/kaggle/working/best_autoencoder.keras')

# Dự đoán trên dữ liệu mới (Giả sử 'new_data' là dữ liệu bạn muốn dự đoán)

predictions = model.predict(X_test1)

# Hiển thị kết quả dự đoán
print(predictions)
mse = np.mean(np.power(X_test1 - predictions,2), axis=1)
err_df = pd.DataFrame({'error': mse, 'truth': y_test1})
# Vẽ ra confusion matrix
threshold = 2.3
y_pred = [1 if e > threshold else 0 for e in err_df.error.values]
conf_matrix = confusion_matrix(err_df.truth, y_pred)

# Vẽ
sns.heatmap(conf_matrix, xticklabels=["Normal", "Fraud"], yticklabels=['Normal','Fraud'], annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predited Values")
plt.ylabel("Truth")
plt.show

# Đánh giá mô hình
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print(" Precision:", precision_score(y_test1, y_pred))
print(" Recall:", recall_score(y_test1, y_pred))
print("AUC-ROC:", roc_auc_score(y_test1, y_pred))