# %% [markdown]
# # Perbandingan Metode K-Nearest Neighbor dan Random Forest dalam Klasifikasi Status Stunting Anak Berdasarkan Data Antropometri
# 
# **Tugas Besar Praktikum Machine Learning**
# 
# Notebook ini membandingkan performa algoritma:
# - **KNN (K-Nearest Neighbor)** - Baseline Model
# - **Random Forest** - Advanced Model
# 
# untuk klasifikasi status stunting anak berdasarkan data antropometri.

# %% [markdown]
# ## 1. Setup Google Colab & Load Data

# %%
# Mount Google Drive (Uncomment jika menggunakan Google Colab)
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

print("✅ Semua library berhasil diimport!")

# %%
# Load Dataset
# Untuk Google Colab, ganti path sesuai lokasi file di Google Drive
# Contoh: '/content/drive/MyDrive/TB Machine Learning/stunting_wasting_dataset.csv'

# Path lokal (untuk testing)
DATA_PATH = 'stunting_wasting_dataset.csv'

# Path Google Colab (uncomment dan sesuaikan)
# DATA_PATH = '/content/drive/MyDrive/TB Machine Learning/stunting_wasting_dataset.csv'

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset berhasil dimuat!")
print(f"📊 Shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

# %%
# Menampilkan 10 data pertama
df.head(10)

# %%
# Info dataset
df.info()

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
# Statistik Deskriptif
print("=" * 60)
print("STATISTIK DESKRIPTIF")
print("=" * 60)
df.describe()

# %%
# Cek Missing Values
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

# %%
# Distribusi Target Variable (Stunting)
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#FFA94D', '#69DB7C', '#4DABF7']
stunting_counts = df['Stunting'].value_counts()
plt.pie(stunting_counts, labels=stunting_counts.index, autopct='%1.1f%%', 
        colors=colors, explode=[0.02]*len(stunting_counts), shadow=True)
plt.title('Distribusi Status Stunting', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nDistribusi Stunting:")
print(stunting_counts)

# %%
# Distribusi Jenis Kelamin
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Jenis Kelamin', palette='Set2')
plt.title('Distribusi Jenis Kelamin', fontsize=14, fontweight='bold')
plt.xlabel('Jenis Kelamin')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# %%
# Distribusi Fitur Numerik
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

features_numeric = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)']
colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']

for idx, (feature, color) in enumerate(zip(features_numeric, colors)):
    axes[idx].hist(df[feature], bins=30, color=color, edgecolor='white', alpha=0.8)
    axes[idx].set_title(f'Distribusi {feature}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frekuensi')

plt.tight_layout()
plt.show()

# %%
# Boxplot berdasarkan Status Stunting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(features_numeric):
    sns.boxplot(data=df, x='Stunting', y=feature, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{feature} vs Stunting', fontsize=12, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
# Heatmap Korelasi
plt.figure(figsize=(8, 6))
# Encode Jenis Kelamin untuk korelasi
df_corr = df.copy()
df_corr['Jenis Kelamin'] = LabelEncoder().fit_transform(df_corr['Jenis Kelamin'])
df_corr['Stunting'] = LabelEncoder().fit_transform(df_corr['Stunting'])
df_corr['Wasting'] = LabelEncoder().fit_transform(df_corr['Wasting'])

correlation = df_corr.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Data Preprocessing

# %%
# Memisahkan fitur dan target
X = df[['Jenis Kelamin', 'Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)']].copy()
y = df['Stunting'].copy()

print("Fitur (X):")
print(X.head())
print(f"\nShape X: {X.shape}")
print(f"\nTarget (y) unique values: {y.unique()}")

# %%
# Label Encoding untuk Jenis Kelamin
le_gender = LabelEncoder()
X['Jenis Kelamin'] = le_gender.fit_transform(X['Jenis Kelamin'])
print(f"Mapping Jenis Kelamin: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# Label Encoding untuk Target (Stunting)
le_stunting = LabelEncoder()
y_encoded = le_stunting.fit_transform(y)
print(f"Mapping Stunting: {dict(zip(le_stunting.classes_, le_stunting.transform(le_stunting.classes_)))}")

# %%
# Feature Scaling menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Data setelah scaling:")
print(X_scaled.describe())

# %%
# Train-Test Split (80:20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Data berhasil dibagi!")
print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Testing set: {X_test.shape[0]} samples")

# %% [markdown]
# ## 4. Model KNN (Baseline)

# %%
# Hyperparameter Tuning - Mencari nilai K optimal
k_values = range(3, 21, 2)  # K = 3, 5, 7, ..., 19
train_scores = []
test_scores = []

print("=" * 60)
print("HYPERPARAMETER TUNING KNN")
print("=" * 60)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    
    print(f"K = {k:2d} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# %%
# Visualisasi K vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'b-o', label='Training Accuracy', linewidth=2, markersize=8)
plt.plot(k_values, test_scores, 'r-s', label='Testing Accuracy', linewidth=2, markersize=8)
plt.xlabel('Nilai K', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: Pengaruh Nilai K terhadap Accuracy', fontsize=14, fontweight='bold')
plt.xticks(k_values)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Menentukan K optimal
best_k = k_values[np.argmax(test_scores)]
print(f"\n🏆 Nilai K optimal: {best_k} dengan Test Accuracy: {max(test_scores):.4f}")

# %%
# Training Model KNN dengan K optimal
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)

# Prediksi
y_pred_knn = knn_model.predict(X_test)

print(f"✅ Model KNN berhasil ditraining dengan K = {best_k}")

# %%
# Evaluasi Model KNN
print("=" * 60)
print("EVALUASI MODEL KNN (BASELINE)")
print("=" * 60)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

print(f"Accuracy : {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall   : {knn_recall:.4f}")
print(f"F1-Score : {knn_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=le_stunting.classes_))

# %%
# Confusion Matrix KNN
plt.figure(figsize=(8, 6))
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_stunting.classes_, yticklabels=le_stunting.classes_)
plt.title('Confusion Matrix - KNN', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Model Random Forest (Advanced)

# %%
# Hyperparameter Tuning menggunakan GridSearchCV
print("=" * 60)
print("HYPERPARAMETER TUNING RANDOM FOREST")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Menggunakan subset data untuk hyperparameter tuning (untuk efisiensi)
X_train_subset = X_train.sample(n=min(10000, len(X_train)), random_state=42)
y_train_subset = y_train[X_train_subset.index]

print("⏳ Melakukan Grid Search (mungkin membutuhkan beberapa menit)...")

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_subset, y_train_subset)

print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
print(f"🏆 Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# %%
# Training Model Random Forest dengan parameter terbaik
rf_model = RandomForestClassifier(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Prediksi
y_pred_rf = rf_model.predict(X_test)

print(f"✅ Model Random Forest berhasil ditraining!")

# %%
# Evaluasi Model Random Forest
print("=" * 60)
print("EVALUASI MODEL RANDOM FOREST (ADVANCED)")
print("=" * 60)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted')
rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1-Score : {rf_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le_stunting.classes_))

# %%
# Confusion Matrix Random Forest
plt.figure(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le_stunting.classes_, yticklabels=le_stunting.classes_)
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Evaluasi dan Perbandingan Model

# %%
# Cross-Validation untuk kedua model
print("=" * 60)
print("CROSS-VALIDATION (5-Fold)")
print("=" * 60)

# Menggunakan subset untuk cross-validation (efisiensi)
X_cv = X_scaled.sample(n=min(20000, len(X_scaled)), random_state=42)
y_cv = y_encoded[X_cv.index]

cv_knn = cross_val_score(KNeighborsClassifier(n_neighbors=best_k), X_cv, y_cv, cv=5, scoring='accuracy')
cv_rf = cross_val_score(RandomForestClassifier(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    random_state=42, n_jobs=-1
), X_cv, y_cv, cv=5, scoring='accuracy')

print(f"KNN Cross-Validation Scores: {cv_knn}")
print(f"KNN Mean CV Score: {cv_knn.mean():.4f} (+/- {cv_knn.std()*2:.4f})")
print()
print(f"Random Forest Cross-Validation Scores: {cv_rf}")
print(f"Random Forest Mean CV Score: {cv_rf.mean():.4f} (+/- {cv_rf.std()*2:.4f})")

# %%
# Tabel Perbandingan Metrik
print("=" * 60)
print("PERBANDINGAN PERFORMA MODEL")
print("=" * 60)

comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean Score'],
    'KNN (Baseline)': [knn_accuracy, knn_precision, knn_recall, knn_f1, cv_knn.mean()],
    'Random Forest (Advanced)': [rf_accuracy, rf_precision, rf_recall, rf_f1, cv_rf.mean()]
}

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.set_index('Metric')

# Format sebagai persentase
df_comparison_display = df_comparison.copy()
for col in df_comparison_display.columns:
    df_comparison_display[col] = df_comparison_display[col].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")

print(df_comparison_display)

# %%
# Visualisasi Perbandingan Metrik
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean Score']
x = np.arange(len(metrics))
width = 0.35

knn_scores = [knn_accuracy, knn_precision, knn_recall, knn_f1, cv_knn.mean()]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1, cv_rf.mean()]

bars1 = ax.bar(x - width/2, knn_scores, width, label='KNN (Baseline)', color='#3498DB', edgecolor='white')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest (Advanced)', color='#2ECC71', edgecolor='white')

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Performa: KNN vs Random Forest', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)

# Menambahkan nilai di atas bar
for bar, score in zip(bars1, knn_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{score:.3f}', ha='center', va='bottom', fontsize=9)
for bar, score in zip(bars2, rf_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{score:.3f}', ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Visualisasi dan Analisis

# %%
# Feature Importance dari Random Forest
plt.figure(figsize=(10, 6))
feature_names = ['Jenis Kelamin', 'Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
plt.bar(range(len(feature_names)), importances[indices], color=colors, edgecolor='white')
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Fitur', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')

for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
    plt.text(i, imp + 0.01, f'{imp:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print("\nFeature Importance Ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# %%
# Perbandingan Confusion Matrix Side by Side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=le_stunting.classes_, yticklabels=le_stunting.classes_)
axes[0].set_title('Confusion Matrix - KNN', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=le_stunting.classes_, yticklabels=le_stunting.classes_)
axes[1].set_title('Confusion Matrix - Random Forest', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Menyimpan Model untuk Deployment

# %%
# Path untuk menyimpan model
# Untuk Google Colab, ganti dengan path Google Drive
# Contoh: '/content/drive/MyDrive/TB Machine Learning/models/'

MODEL_PATH = 'models/'

# Untuk Google Colab (uncomment dan sesuaikan)
# MODEL_PATH = '/content/drive/MyDrive/TB Machine Learning/models/'

# %%
# Menyimpan Model KNN
joblib.dump(knn_model, f'{MODEL_PATH}knn_model.joblib')
print(f"✅ Model KNN berhasil disimpan ke: {MODEL_PATH}knn_model.joblib")

# Menyimpan Model Random Forest
joblib.dump(rf_model, f'{MODEL_PATH}rf_model.joblib')
print(f"✅ Model Random Forest berhasil disimpan ke: {MODEL_PATH}rf_model.joblib")

# Menyimpan Scaler
joblib.dump(scaler, f'{MODEL_PATH}scaler.joblib')
print(f"✅ Scaler berhasil disimpan ke: {MODEL_PATH}scaler.joblib")

# Menyimpan Label Encoder untuk Jenis Kelamin
joblib.dump(le_gender, f'{MODEL_PATH}label_encoder_gender.joblib')
print(f"✅ Label Encoder Gender berhasil disimpan ke: {MODEL_PATH}label_encoder_gender.joblib")

# Menyimpan Label Encoder untuk Stunting
joblib.dump(le_stunting, f'{MODEL_PATH}label_encoder_stunting.joblib')
print(f"✅ Label Encoder Stunting berhasil disimpan ke: {MODEL_PATH}label_encoder_stunting.joblib")

# %%
# Menyimpan informasi model (untuk referensi)
model_info = {
    'knn_best_k': best_k,
    'rf_best_params': grid_search.best_params_,
    'feature_names': feature_names,
    'stunting_classes': list(le_stunting.classes_),
    'gender_classes': list(le_gender.classes_),
    'knn_metrics': {
        'accuracy': knn_accuracy,
        'precision': knn_precision,
        'recall': knn_recall,
        'f1_score': knn_f1
    },
    'rf_metrics': {
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1_score': rf_f1
    }
}

joblib.dump(model_info, f'{MODEL_PATH}model_info.joblib')
print(f"✅ Model info berhasil disimpan ke: {MODEL_PATH}model_info.joblib")

print("\n" + "=" * 60)
print("SEMUA MODEL BERHASIL DISIMPAN!")
print("=" * 60)

# %% [markdown]
# ## 9. Kesimpulan

# %%
print("=" * 60)
print("KESIMPULAN")
print("=" * 60)

print(f"""
📊 RINGKASAN HASIL PERBANDINGAN
{'=' * 60}

1. MODEL KNN (BASELINE)
   - Nilai K optimal: {best_k}
   - Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)
   - Precision: {knn_precision:.4f}
   - Recall: {knn_recall:.4f}
   - F1-Score: {knn_f1:.4f}
   - CV Mean Score: {cv_knn.mean():.4f}

2. MODEL RANDOM FOREST (ADVANCED)
   - Parameters: {grid_search.best_params_}
   - Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)
   - Precision: {rf_precision:.4f}
   - Recall: {rf_recall:.4f}
   - F1-Score: {rf_f1:.4f}
   - CV Mean Score: {cv_rf.mean():.4f}

{'=' * 60}
""")

# Menentukan model terbaik
if rf_accuracy > knn_accuracy:
    better_model = "Random Forest"
    improvement = ((rf_accuracy - knn_accuracy) / knn_accuracy) * 100
    print(f"🏆 MODEL TERBAIK: {better_model}")
    print(f"   Peningkatan akurasi dibandingkan KNN: {improvement:.2f}%")
else:
    better_model = "KNN"
    print(f"🏆 MODEL TERBAIK: {better_model}")

print(f"""
💡 ANALISIS:

KELEBIHAN KNN:
- Mudah diimplementasikan dan dipahami
- Tidak memerlukan training time yang lama
- Cocok untuk dataset kecil hingga menengah

KEKURANGAN KNN:
- Lambat pada saat prediksi untuk dataset besar
- Sensitif terhadap outlier dan skala data
- Perlu menentukan nilai K yang optimal

KELEBIHAN RANDOM FOREST:
- Dapat menangani hubungan non-linear
- Robust terhadap overfitting
- Memberikan informasi feature importance
- Performa umumnya lebih baik untuk dataset besar

KEKURANGAN RANDOM FOREST:
- Lebih kompleks dan membutuhkan resource lebih besar
- Training time lebih lama
- "Black box" - sulit diinterpretasi

📌 REKOMENDASI:
Berdasarkan hasil evaluasi, model {better_model} direkomendasikan untuk
klasifikasi status stunting karena memberikan performa yang lebih baik
pada dataset ini.

Model telah disimpan dan siap untuk deployment ke aplikasi Streamlit!
""")
