import imaplib
import email
from email.header import decode_header
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Gmail bilgileri
EMAIL = "example@gmail.com"
PASSWORD = "lxmeqjvqzrgxecur"  # uygulama şifresi

# Model ve vectorizer yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Gmail'e bağlan
imap = imaplib.IMAP4_SSL("imap.gmail.com")
imap.login(EMAIL, PASSWORD)
imap.select("inbox")

# Tüm mailleri al
status, messages = imap.search(None, "ALL")
mail_ids = messages[0].split()

mail_bodies = []

for mail_id in reversed(mail_ids):
    status, msg_data = imap.fetch(mail_id, "(RFC822)")
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_dispo = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_dispo:
                        try:
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        except:
                            body = ""
                        break
            else:
                try:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                except:
                    body = ""
            mail_bodies.append(body)

imap.logout()

# TF-IDF ile vektörleştir
X = vectorizer.transform(mail_bodies)
preds = model.predict(X)

# Tahminleri yazdır
for body, label in zip(mail_bodies, preds):
    status = "SPAM" if label == 1 else "NORMAL"
    print(f"İçerik: {body[:80]}\nTahmin: {status}\n")
