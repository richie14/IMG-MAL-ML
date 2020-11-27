import binascii
file1= open("test.txt","r+") 
data=file1.read()
data=data.replace(' ', '')
data=data.replace('\n', '')
data = binascii.a2b_hex(data)
with open('hextoimg.jpg', 'wb') as image_file:
    image_file.write(data)   
file1.close() 
