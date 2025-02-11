import os
import sys
import json
import cv2
import requests
import image_to_base64

def get_data(json_data, url):
    """
    Sends JSON data to the FastAPI application and returns the response.

    :param json_data: The JSON data to send.
    :param url: The URL of the FastAPI application's endpoint.
    :return: Parsed JSON response or None if the request failed.
    """
    try:
        response = requests.post(url, json=json_data)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()  # Return parsed JSON response
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending the request: {e}")
        return None

def run(input_json):
    server_url = 'http://192.168.134.252:8001/anpr'
    
    print("Processing data...")
    output = get_data(json_data=input_json, url=server_url)
    
    if output is not None:
        print("\nOutput from the application:")
        
        op = json.loads(json.dumps(output, indent=4))  # Pretty print the output

        lp_bbox = op['result']['lp_bbox']
        lp_np = op['result']['license_plate_number']
        
        with open('code_snippet.txt', 'a') as file:
            file.write(f'{lp_np} \n')

        return lp_bbox, lp_np

    else:
        print("No output received.")

def get_json(image_bright, image_dark, image_height, image_width):
    """
    Constructs the JSON payload to be sent to the FastAPI application.

    :param image_bright: Base64 encoded bright image.
    :param image_dark: Base64 encoded dark image.
    :param image_height: Height of the image.
    :param image_width: Width of the image.
    :return: A dictionary representing the JSON payload.
    """
    return {
        "object-id": 1,
        "vehicle-id": 2,
        "image_bright": image_bright,
        "image_dark": image_dark,
        "bbox": [0, 0, image_width, image_height],
        "lpd": True,
        "lp_enhance": True,
        "vehicle_enhance": True,
        "vehicle_color": True,
        "lpr": True,
        "make_model": True,
        "embeddings": False
    }

def process_images(image, fov_img):
    """
    Processes all images in the specified folder and sends them to the FastAPI application.

    :param input_folder: Path to the folder containing images.
    """
    # for image_name in os.listdir(input_folder):
    #     image_path = os.path.join(input_folder, image_name)
    # 

    image_path = image


    # # Read and process the image
    # cv2_image = cv2.imread(image_path)

    # if cv2_image is None:
    #     print(f"Could not read image: {image_path}. Skipping.")
    #     # continue

    height, width, _ = image_path.shape

    image_bright = image_to_base64.image_to_base64(image_path)
    image_dark = image_to_base64.image_to_base64(image_path)

    input_json = get_json(image_bright=image_bright, image_dark=image_dark,
                            image_height=height, image_width=width)
    
    lp_bbox, lp_np = run(input_json=input_json)

    if lp_np is not None:

        center_x, center_y, width, height = map(float, lp_bbox)

        # Calculate top-left corner coordinates
        x = int(center_x - width / 2)
        y = int(center_y - height / 2)
        
        cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), (255, 0, 0), 2)  # Red rectangle
        cv2.putText(image, lp_np, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2)  # White text


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 image_folder_client.py <image_folder>")
        sys.exit(1)  # Exit with an error code
    
    input_folder = sys.argv[1]

    if not os.path.isdir(input_folder):
        print(f"The specified path is not a directory: {input_folder}")
        sys.exit(1)  # Exit with an error code
    
    process_images(input_folder)

if __name__ == '__main__':
    main()
