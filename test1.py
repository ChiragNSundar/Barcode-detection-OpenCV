import cv2
import numpy as np
import csv
import pyzbar.pyzbar as pyzbar

# Function to decode QR codes
def decode_qr_codes(image, csv_writer, existing_rows):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find QR codes in the image
    decoded_objects = pyzbar.decode(gray)

    # Process each decoded object
    for obj in decoded_objects:
        # Extract the data and type of the QR code
        data = obj.data.decode("utf-8")
        code_type = obj.type

        # Draw a rectangle around the QR code
        cv2.rectangle(image, (obj.rect.left, obj.rect.top), (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height), (255, 0, 0), 2)

        # Display the QR code data
        cv2.putText(image, f"{code_type}: {data}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # If the detected code is a QR code, display a side box with the link
        if code_type == "QRCODE":
            # Extract the link from the QR code data
            link = extract_link_from_qr_code(data)

            # Draw a side box with the link
            cv2.rectangle(image, (10, 100), (300, 150), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, f"Link: {link}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Write the QR code data to the CSV file if it hasn't been written before
            write_to_csv(['QRCODE', data, link], csv_writer, existing_rows)

    return image

# Function to extract the link from QR code data
def extract_link_from_qr_code(qr_code_data):
    # Extract the link from the QR code data
    # You can use regular expressions or any other method to extract the link
    # link = re.search("(?P<url>https?://[^\s]+)", qr_code_data).group("url")
    # Here, we assume that the QR code data is already a link
    link = qr_code_data
    return link

# Function to write data to CSV if it doesn't already exist and is not empty
def write_to_csv(row, csv_writer, existing_rows):
    # Skip if any field in the row is empty
    if any(not field for field in row):
        return

    # Write the new row if it doesn't already exist
    if row not in existing_rows:
        csv_writer.writerow(row)
        existing_rows.append(row)

# Load the barcode detection model
barcode_detector = cv2.barcode_BarcodeDetector()

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize the CSV file
csv_file = open('barcode_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Barcode Type', 'Barcode/QR Code Data', 'Link'])
existing_rows = [['Barcode Type', 'Barcode/QR Code Data', 'Link']]

# Function to resize frame to fit the window
def resize_frame(frame, window_name):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_width = cv2.getWindowImageRect(window_name)[2]
    new_height = int(new_width / aspect_ratio)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Set the window name
window_name = 'Barcode and QR Code Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect and decode barcodes
        success, decoded_info, decoded_points = barcode_detector.detectAndDecode(frame)

        if success and decoded_info and any(decoded_info):
            # Draw rectangles around the detected barcodes
            if decoded_points is not None:
                for p in decoded_points:
                    points = np.int32(p).reshape(-1, 2)
                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Display the detected barcode type and data
            for info in decoded_info:
                if info:  # Check if info is not empty
                    cv2.putText(frame, f"Barcode Data: {info}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # Write the barcode data to the CSV file if it hasn't been written before
                    write_to_csv(['Barcode', info, ''], csv_writer, existing_rows)

        # Decode QR codes
        frame = decode_qr_codes(frame, csv_writer, existing_rows)

        # Resize the frame to fit the window
        frame = resize_frame(frame, window_name)

        # Display the output
        cv2.imshow(window_name, frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# Release the webcam and close the CSV file
cap.release()
csv_file.close()
cv2.destroyAllWindows()
