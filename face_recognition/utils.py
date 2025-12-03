import base64

def decode_base64_image(base64_string, filepath):
    image_data = base64.b64decode(base64_string.split(',')[1])
    with open(filepath, 'wb') as f:
        f.write(image_data)
