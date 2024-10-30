# coding: UTF-8
"""
For string encryption/decryption functions.

Dependency:
    cryptography (3.4),

last edited on 2021.10.27.
"""
import secrets
from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

DEBUG = False

#-------------------------------------------------------------------------------
def deriveSecretKey(password, salt, iterations):
    """ Derive a secret key from a given password and salt

    Args:
        password (bytes).
        salt (bytes).
        iterations (int).

    Returns:
        (bytes): Derived key.
    """
    if DEBUG: print("modEncryption.passwordEncrypt()")

    backend = default_backend()
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), 
                     length=32, 
                     salt=salt, 
                     iterations=iterations, 
                     backend=backend)
    return b64e(kdf.derive(password))

#-------------------------------------------------------------------------------

def passwordEncrypt(message, password, iterations=100000):
    """ Encrypt (symmetric encryption) message using password

    Args:
        message (bytes).
        password (str).
        iterations (int).

    Returns:
        (bytes): Encrypted message.
    """
    if DEBUG: print("modEncryption.passwordEncrypt()")

    salt = secrets.token_bytes(16)
    key = deriveSecretKey(password.encode(), salt, iterations)
    return b64e(b'%b%b%b'%(salt, 
                           iterations.to_bytes(4, 'big'), 
                           b64d(Fernet(key).encrypt(message))
                           )
                )

#-------------------------------------------------------------------------------

def passwordDecrypt(token, password):
    """ Decrypt the encrypted (symmetric encryption) token using password

    Args:
        token (bytes).
        password (str).

    Returns:
        (str): Decrypted token.
    """
    if DEBUG: print("modEncryption.passwordDecrypt()")

    decoded = b64d(token)
    salt, iter, token = decoded[:16], decoded[16:20], b64e(decoded[20:])
    iterations = int.from_bytes(iter, 'big')
    key = deriveSecretKey(password.encode(), 
                          salt, 
                          iterations)
    return Fernet(key).decrypt(token)

#-------------------------------------------------------------------------------


