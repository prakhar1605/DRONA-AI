import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Credentials Load karo
cred = credentials.Certificate("backend/firebase_credentials.json")
firebase_admin.initialize_app(cred)

print("✅ Firebase Initialized Successfully!")
