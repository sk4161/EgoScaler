class CameraConfig:
    fps = 20.0 # sampling rate 
    time_window = 2.0 # action time span [-2, 2]

    class Devices:
        class Aria:
            pinhole_image_size = 1408 # image width and height
            focal_len = 605.343
            principal_point = 703.5
            
        aria = Aria()

    devices = Devices()
