import base64
import cv2

# def image_to_base64(image_path):
#     """
#     Convert an image to a Base64-encoded string.
#     """
#     with open(image_path, "rb") as image_file:
#         # Read the image file
#         image_data = image_file.read()
#         # Encode the image data to Base64

#         base64_encoded = base64.b64encode(image_data)
#         # Decode bytes to string
#         base64_string = base64_encoded.decode('utf-8')
        
#         return base64_string

def image_to_base64(image_array):
    """
    Convert an image to a Base64-encoded string.
    """
    
    # Encode the image as a PNG (you can use other formats like JPEG)
    success, encoded_image = cv2.imencode('.png', image_array)
    
    if not success:
        raise Exception("Could not encode image")
    
    # Convert the encoded image to bytes
    image_data = encoded_image.tobytes()

    base64_encoded = base64.b64encode(image_data)
    # Decode bytes to string
    base64_string = base64_encoded.decode('utf-8')
    
    return base64_string
