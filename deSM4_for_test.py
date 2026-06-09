from gmssl import sm4, func
import binascii
SM4_KEY = b'1234567890abcdef'
sm4_crypt = sm4.CryptSM4()
def sm4_encrypt(plain_text):
    sm4_crypt.set_key(SM4_KEY, sm4.SM4_ENCRYPT)
    plain_bytes = plain_text.encode('utf-8')
    encrypt_bytes = sm4_crypt.crypt_ecb(plain_bytes)
    return binascii.hexlify(encrypt_bytes).decode('utf-8')
print(sm4_encrypt("admin"))