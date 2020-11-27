import binascii
import sys
filename = 'test.jpg'
with open(filename, 'rb') as f:
    content = f.read()
sys.stdout = open("test.txt", "w")
print(binascii.hexlify(content))
sys.stdout.close()
