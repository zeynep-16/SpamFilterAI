import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Eğitimde kaydedilen model ve vectorizer'ı yüklüyoruz
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Tahmin etmek istenilen metinleri buraya giriyoruz
mails = [
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "Toplantımız yarın saat 14:00'te başlayacak.",
    "URGENT: Your account is compromised. Act now!"
]

# Metinleri vektörleştirme
X = vectorizer.transform(mails)

# Tahmin yapma
preds = model.predict(X)

# Sonuçları yazdırma
for mail, label in zip(mails, preds):
    status = "SPAM" if label == 1 else "NORMAL"
    print(f"Konu: {mail}\nTahmin: {status}\n")
