# Pedestrian-Navigation-without-GPS
Final Project of our Electrical Engineering degree.

# Indoor 3D Movement Tracking and Visualization

This project detects and classifies the movements of a person walking indoors (step forward/backward, crab step left/right, turn left/right, stairs up/down, elevator up/down) using the accelerometer, gyroscope, barometer and camera of a chest-mounted mobile phone, with no GPS. The output is a CSV file with the classified movements and a 3D visualization of the walked path.

## Prerequisites

1. **Android Device with IP Webcam**: Install the IP Webcam app from the Google Play Store. The phone needs a barometer for the stairs and elevator detection.
2. **Chest holder** for the phone. All the thresholds assume it, so this is not optional.
3. **Python 3.8** and the libraries listed in `requirements.txt`:
    - `numpy`
    - `pandas`
    - `scipy`
    - `matplotlib`
    - `requests`
    - `opencv-python`
    - `pyttsx3`

    Install them with:
    ```bash
    pip install -r requirements.txt
    ```

## Setup Instructions

1. **Mounting the Phone**:
    - Put the phone in the chest holder in **portrait** orientation, with the screen facing forward, so the rear camera looks ahead.
    - Keep it in that position for the whole walk. If it is rotated in the middle of a run the axes swap and the detection breaks.

2. **Mobile Device Connection**:
    - Either connect the phone to the PC with a USB cable and turn on **USB tethering** (Settings > Network and internet > Hotspot and tethering), or put the phone and the PC on the same WiFi network. USB tethering is recommended because it keeps working in elevators and stairwells.
    - Open the IP Webcam app and start the server.
    - Note the IP address. On WiFi it is shown on the app screen (e.g. `http://192.168.0.101:8080/`). On USB tethering, run `ipconfig` on the PC and take the **Default Gateway** of the **Remote NDIS** adapter (e.g. `10.194.198.210`).
    - Open `config.py` and set that address in the `PHONE_IP` variable.
    - Check the connection before walking:
      ```bash
      python ipwebcam_client.py
      ```

3. **Step Length Estimation**:
    - Measure a straight path of about 10 metres, then run:
      ```bash
      python calibrate_step_length.py
      ```
    - Enter your ID and the length of the path, walk it with forward steps only, and press Enter when you reach the end. The step length is saved in the `user_step_sizes.json` file.

4. **Running the Main Program**:
    - Make sure `PHONE_IP` in `config.py` matches the IP from the IP Webcam app, and that the server is running on the phone.
    - Run the main program:
      ```bash
      python main_ipwebcam.py
      ```
    - Enter the same ID you used for the step length estimation.
    - The live 3D map and the movement signal open in two windows. Walk the path, and press Enter in the console when you are done.

5. **Output**:
    - `data.csv`, the walked path with the movement type of every point.
    - `fig1_3d_pdr_map.png`, the 3D visualization of the trajectory, coloured by movement type, plus `fig2` to `fig7` with the filtered signals, the detected movements and turns, and the barometer elevation.
    - `accel_log.csv` and `pressure_log.csv`, the raw sensor recordings of the run.

## Code Overview

- **main_ipwebcam.py**: The main program. Reads the sensors and the camera, detects and classifies the movements, updates the position and saves the results.
- **config.py**: All the thresholds and settings of the system.
- **ipwebcam_client.py**: Reads the sensors and the camera frames from the IP Webcam app.
- **signal_utils.py**: Butterworth filtering of the accelerometer and the gyroscope.
- **movement_detection.py**: Peak detection for steps and crab steps.
- **turn_detection.py**: Turn detection from the gyroscope.
- **optical_flow.py**: Decides forward or backward from the camera, by measuring whether the scene expands or contracts.
- **pressure_utils.py**: Converts the barometric pressure to relative altitude.
- **mapping.py**: Updates the position and the heading of the walker.
- **plotting.py**: The live windows and the saved figures.
- **audio_feedback.py**: Announces the detected movements out loud during the run.
- **calibrate_step_length.py**: Estimates the step length of one person.
- **buffer_utils.py**, **keypress_stop.py**, **step_size_store.py**: Rolling buffers, the Enter-to-stop listener, and the per-user step length file.

## Important Notes

- The IP address in `config.py` must be updated according to your mobile device's IP Webcam address. With USB tethering, Android gives a new address every time it is turned on, so it changes between sessions.
- Run `calibrate_step_length.py` first to estimate the step length, which is then used in every run of `main_ipwebcam.py`. Without an ID, the default step length of 0.70 m is used.
- The movement types printed during the walk are not always final. A flight of stairs and the direction of a stretch of walking are only clear once the whole recording exists, so the path is computed again when the run stops, and that is the version saved to `data.csv` and drawn in the figures. Every correction is listed in the summary.
- `main_ipwebcam.py` saves the raw accelerometer and pressure recordings on the user's PC every few seconds, so a run is not lost if the program is stopped.
- The voice feedback needs the Windows speech engine. Set `USE_AUDIO = False` in `config.py` to run silently.

## Example Workflow

1. Mount the phone on the chest, connect it and start the IP Webcam server.
2. Set `PHONE_IP` in `config.py` and check the connection:
    ```bash
    python ipwebcam_client.py
    ```
3. Estimate step length:
    ```bash
    python calibrate_step_length.py
    ```
4. Run the main detection program:
    ```bash
    python main_ipwebcam.py
    ```
5. Walk the path, press Enter to stop, and look at `data.csv` and `fig1_3d_pdr_map.png`.

## Troubleshooting

- If nothing arrives from the phone, check that the IP Webcam server is started, that the IP in `config.py` matches the one the app or `ipconfig` shows, and that the phone answers a `ping`.
- If the run stops in the middle, you were probably on WiFi and walked out of range. Use USB tethering instead.
- If steps are missed, lower `FORWARD_STEP_HEIGHT` in `config.py`. If steps are counted that were not taken, raise it. Check the holder is tight first, because a loose phone bounces on its own.
- If crab steps are detected as forward steps, raise `CRAB_RATIO_TO_FORWARD`.
- If left and right come out reversed, flip `POSITIVE_GYRO_Y_IS_LEFT` for turns or `CRAB_LEFT_IS_POSITIVE_X` for crab steps.
- Short flights of stairs are not detected. The barometer noise is about the size of one stair, so only a flight of roughly four stairs or more registers.
- Make sure the mobile device is securely mounted on the chest in portrait orientation for accurate data capture.
