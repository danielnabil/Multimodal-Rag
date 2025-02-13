from PIL import Image
from io import BytesIO
import base64

def decode_base64_image(b64_string):
    try:
        img_data = base64.b64decode(b64_string)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None