import firebase_admin
from firebase_admin import credentials, auth

# Firebase Credentials Load karo
cred = credentials.Certificate("backend/firebase_credentials.json")  # Path check kar lo!
firebase_admin.initialize_app(cred)

print("✅ Firebase Authentication Initialized!")
