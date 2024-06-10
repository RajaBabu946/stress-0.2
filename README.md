import cv2
import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=np.RankWarning)
# Initialize variables for ROI selection and image capture
roi_selected = True
roi_start = (497,475)  # Example coordinates for the top-left corner of ROI
roi_end = (1256,833)    # Example coordinates for the bottom-right corner of ROI
capture_images = False
image_count = 0

# Initialize the camera capture object
#camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 indicates the default camera, change if needed

# Video file path
video_file = "C:\\Users\\rajab\\Pictures\\Camera Roll\\0.3 original.mp4"

# Initialize the video capture object
camera = cv2.VideoCapture(video_file)
# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the camera's resolution to the calibrated resolution if applicable
# Replace <calibrated_width> and <calibrated_height> with your actual values
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Replace with your calibrated width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Replace with your calibrated height

# Output directory for saving images
output_directory = r"C:\Users\rajab\Pictures\Camera Roll\a"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Time duration for capturing images (4 minutes and 12 seconds)
capture_duration = 252  # 252 seconds

# Lists to store stress values and corresponding time points
stress_values = []
time_points = []

# Initialize Excel File and DataFrame for data logging
excel_file = os.path.join(output_directory, "stress_data.xlsx")
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=['Time', 'Stress_MPa'])
    df.to_excel(excel_file, index=False)

# Initialize a figure for the real-time graph
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Internal Stress (MPa)')
plt.title('Internal Stress vs. Time')
line, = ax.plot([], [])

# Create a window to display the coordinates
cv2.namedWindow("Coordinates")

# Create the "Frame" window and the trackbar
def nothing(x):
    pass

cv2.namedWindow("Frame")
cv2.createTrackbar("quality", "Frame", 1, 100, nothing)
cv2.setTrackbarPos("quality", "Frame", 19)  # Set quality to 38

# Main loop to capture frames and process data
while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Draw ROI rectangle on the frame
    cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)

    # Display the frame with the ROI highlighted
    cv2.imshow("Frame with ROI", frame)

    # Display the frame in the "Frame" window
    cv2.imshow("Frame", frame)

    # Handle keyboard input
    key = cv2.waitKey(1)

    if key == 27:  # Esc key pressed
        break

        # Release the camera and close all windows
        camera.release()
        cv2.destroyAllWindows()

# Stoney Formula Constants
E_substrate = 120655  # Modulus of elasticity of the substrate in kg/cm²
T_substrate = 0.05077  # Thickness of the substrate in millimeters
L_substrate = 76.2  # Length of substrate in millimeters
t_deposit = 0.002538  # Deposit average thickness in millimeters
M = 1.714  # Correction factor for modulus of elasticity difference

# Number of points to use for the moving average filter
moving_avg_window = 2500


# Function to calculate distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate moving average
def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


# Function to remove spikes from the array
def despike(yi, th=1.e-8):
    y = np.copy(yi)
    n = len(y)

    if n == 0:
        return y

    x = np.arange(n)
    c = np.argmax(y)
    d = abs(np.diff(y))

    try:
        l = c - 1 - np.where(d[c - 1::-1] < th)[0][0]
        r = c + np.where(d[c:] < th)[0][0] + 1
    except:  # no spike, return unaltered array
        return y

    # for fit, use area twice wider than the spike
    if (r - l) <= 3:
        l -= 1
        r += 1
    s = int(round((r - l) / 2.))
    lx = l - s
    rx = r + s

    # make a gap at spike area
    xgapped = np.concatenate((x[lx:l], x[r:rx]))
    ygapped = np.concatenate((y[lx:l], y[r:rx]))

    if len(xgapped) > 0:
        # quadratic fit of the gapped array
        try:
            z = np.polyfit(xgapped, ygapped, 1)
            p = np.poly1d(z)
            y[l:r] = p(x[l:r])
        except np.linalg.LinAlgError:
            # Handle the case where the polynomial fit fails
            pass

    return y


# Create a timer to track the capture duration
start_time = time.time()

while True:
    ret, frame = camera.read()

    if roi_selected:
        frame_roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    quality = cv2.getTrackbarPos("quality", "Frame")
    quality = quality / 100.0 if quality > 0 else 0.01

    if roi_selected:
        corners = cv2.goodFeaturesToTrack(gray, 100, quality, 20)
    else:
        corners = None

    if corners is not None:
        corners = np.intp(corners)
        
        leftmost_corner = corners[corners[:, :, 0].argmin()][0] + roi_start

        max_y = corners[corners[:, :, 1].argmax()][0][1]
        bottommost_corners = [tuple(c[0]) for c in corners if c[0][1] == max_y]

        for i, bottommost_corner in enumerate(bottommost_corners):
            x, y = bottommost_corner

            if roi_selected:
                x += roi_start[0]
                y += roi_start[1]

            center_x = (roi_start[0] + roi_end[0]) // 2
            center_y = (roi_start[1] + roi_end[1]) // 2

            distance_pixels = calculate_distance((x, y), (center_x, center_y))
            distance_cm = (distance_pixels / 100) * 2
            distance_inches = distance_cm / 2.54  # Convert cm to inches
            distance_mm = distance_inches / 2 * 25.4  # Convert inches to mm
            # distance_mm = distance_cm * 10

            stress_MPa = (E_substrate * (T_substrate ** 2) * M * distance_mm/2) / (3 * (L_substrate ** 2) * t_deposit)
            stress_PSI = stress_MPa * 145

            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
            cv2.putText(frame, f"Bottommost Point {i + 1}: X = {x}, Y = {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(frame, f"Center Point: X = {center_x}, Y = {center_y}", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"Distance to Center (pixels): {distance_pixels:.2f}", (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Distance to Center (cm): {distance_cm:.2f}", (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Distance to Center (mm): {distance_mm:.2f}", (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Deposit Stress (MPa): {stress_MPa:.2f}", (x, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            cv2.putText(frame, f"Deposit Stress (PSI): {stress_PSI:.2f}", (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

        elapsed_time = int(time.time() - start_time)
        if elapsed_time <= 252:
            stress_values = np.append(stress_values, stress_MPa)
            time_points = np.append(time_points, elapsed_time)

            # Remove spikes from the stress values
            if len(stress_values) >= 2:
                stress_values = despike(stress_values)

            # Apply moving average filter to smooth the stress values
            if len(stress_values) >= moving_avg_window:
                smoothed_stress_values = moving_average(np.array(stress_values), moving_avg_window)

                # Update the real-time graph with the smoothed stress values
                line.set_xdata(time_points[-len(smoothed_stress_values):])
                line.set_ydata(smoothed_stress_values)
                ax.relim()
                ax.autoscale_view()

            # Append data to DataFrame and Excel file only if the number of rows is less than or equal to 252
            if len(time_points) <= 252:
                df = pd.read_excel(excel_file)
                new_data = pd.DataFrame({'Time': [elapsed_time], 'stress_MPa': [stress_MPa]})
                df = pd.concat([df, new_data], ignore_index=True)
                df = df.drop_duplicates(subset=['Time'])  # Drop duplicate entries based on Time column
                df.to_excel(excel_file, index=False)

        if roi_selected and elapsed_time >= 0 and not capture_images:
            capture_images = True

        if capture_images and image_count < capture_duration:
            if int(elapsed_time) > image_count:
                image_count += 1
                image_filename = os.path.join(output_directory, f"image_{image_count}.png")
                cv2.imwrite(image_filename, frame)

        if image_count >= capture_duration:
            print(f"Image capture completed.")
            break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if elapsed_time >= 252:
        print(f"Data collection completed for 252 seconds.")
        break

camera.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
