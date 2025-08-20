import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# CSV dosyasını oku
df = pd.read_csv("spam.csv", encoding='latin-1')

# Sütunları yeniden adlandır 
df = df[['Category', 'Message']]
df.columns = ['label', 'text']

# Etiketleri sayıya çevir: spam = 1, ham = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Eğitim, validation ve test verilerini ayırıyoruz
X_train, X_temp, y_train, y_temp = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Validation ve test verilerini ayıralım
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Gelişmiş TF-IDF vektörleştirici
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=1
)

# Metinleri vektörleştir
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Lojistik regresyon modelini oluştur
model = LogisticRegression(max_iter=1000)

# Modeli eğitim verisiyle eğit
model.fit(X_train_vec, y_train)

# Validation verisi ile performansı kontrol et
val_preds = model.predict(X_val_vec)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test verisiyle son değerlendirme
y_pred = model.predict(X_test_vec)
print(f"Doğruluk: {accuracy_score(y_test, y_pred):.4f}")
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Model ve vectorizer'ı kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Eğitim tamamlandı, model ve vectorizer kaydedildi.")
