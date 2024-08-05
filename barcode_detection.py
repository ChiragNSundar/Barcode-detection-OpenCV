import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

# Function to decode QR codes
def decode_qr_codes(image):
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

    return image

# Function to extract the link from QR code data
def extract_link_from_qr_code(qr_code_data):
    # Extract the link from the QR code data
    # You can use regular expressions or any other method to extract the link
    # For example, if the QR code data is a URL, you can extract it using regex
    # link = re.search("(?P<url>https?://[^\s]+)", qr_code_data).group("url")
    # Here, we assume that the QR code data is already a link
    link = qr_code_data

    return link

# Load the barcode detection model
barcode_detector = cv2.barcode.BarcodeDetector()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect and decode barcodes
        decoded_frame = barcode_detector.detectAndDecode(frame)

        if decoded_frame:
            # Draw rectangles around the detected barcodes
            if decoded_frame[2]:
                for p in decoded_frame[2]:
                    cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (0, 255, 0), 2)

            # Display the detected barcode type and data
            cv2.putText(frame, f"Barcode Type: {decoded_frame[1]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Barcode Data: {decoded_frame[0]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Decode QR codes
        frame = decode_qr_codes(frame)

        # Display the output
        cv2.imshow('Barcode and QR Code Detection', frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# Release the webcam
cap.release()
cv2.destroyAllWindows()