from ultralytics import YOLO
import cv2
import os
from image_folder_client import process_images

# Load the model
model = YOLO('/home/user/srihari/sk_data_preparation/best5.pt')

# Define the source video
source = '/home/user/srihari/rlvds_lpr'

# List of all class indices
all_class_indices = list(range(25))  # Total 25 classes based on your list

# Indices of the violation classes including HELMET and BACK_HELMET
violation_classes = ['person_nsb','no_helmet','phone','phone_right','phone_left','cap']  # Add indices for HELMET and BACK_HELMET
additional_classes = ['car','truck','bus','auto','ped','bicycle','Mini-truck','Mini-bus','two-wheeler','person_sb']  # HELMET and BACK_HELMET indices

classes_list = ['car','person_sb','person_nsb','helmet','no_helmet','phone','no_phone','tr','notr','truck','bus','auto','ped','bicycle','phone_right','phone_left',
'Mini-truck','Mini-bus','two-wheeler','back_helmet','back_no_helmet','cap','back_cap','half_helmet','back_half_helmet','sb_not_clear']

# Index of "TWO_WHEELER"
two_wheeler_index = 17

# Exclude the "TWO_WHEELER" index
class_indices = [i for i in all_class_indices if i != two_wheeler_index]

output_file = '/home/user/srihari/rlvds_lpr.avi'
cap = cv2.VideoCapture(output_file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Ensure output directory exists
output_dir = "/home/user/srihari/rlvds_lpr_out"
os.makedirs(output_dir, exist_ok=True)

# Run the prediction
results = model.predict(source=source, save=False, imgsz=640, show_conf=False, show_labels=False, stream=True, classes=class_indices)

# Process and filter results
for result in results:
    frame = result.orig_img
    
    for det in result.boxes:
        # Get the class index of the detected object
        class_id = int(det.cls)
        
        # Convert bbox tensor to numpy array and extract coordinates
        bbox = det.xyxy.cpu().numpy().flatten()  # Convert tensor to numpy array and flatten it
        conf = float(det.conf)  # Convert tensor to float
        label = f'{model.names[class_id]}'
        print("-", label, "-", sep=None)
        
        xmin, ymin, xmax, ymax = bbox
        cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        cv2.imwrite("./cropped_image.jpg", cropped_image)
        
        #get anpr
        if ymin < 800 < ymax:
            process_images(cropped_image, frame)

        frame[int(ymin):int(ymax), int(xmin):int(xmax)] = cropped_image
        
        if label in additional_classes:
            print("--", label, "--")
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif label == "tr":
            print("++", label, "++")
            cv2.rectangle(frame, (int(bbox[0] + 5), int(bbox[1] + 5)), (int(bbox[2] - 5), int(bbox[3] - 5)), (50, 50, 255), 2)
            cv2.putText(frame, label, (int(bbox[0] + 5), int(bbox[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
        elif label in violation_classes:
            print("++", label, "++")
            cv2.rectangle(frame, (int(bbox[0] + 5), int(bbox[1] + 5)), (int(bbox[2] - 5), int(bbox[3] - 5)), (50, 50, 255), 2)
            cv2.putText(frame, label, (int(bbox[0] + 5), int(bbox[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
    
    # Save the output image with the same name as the input image
    input_image_name = os.path.basename(result.path)  # Get the input image name
    output_image_path = os.path.join(output_dir, input_image_name)  # Generate output path
    cv2.imwrite(output_image_path, frame)  # Save the frame as an image

    # Write the processed frame to the video output
    out.write(frame)

# Release resources
cap.release()
out.release()
print(f'Filtered output video saved as {output_file}')
