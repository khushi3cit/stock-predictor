# hash_passwords.py
import streamlit_authenticator as stauth

passwords = ['pass123', 'secure456']
hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)
