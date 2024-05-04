"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 30

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()
    angle = 0.0
    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep
    
    # Processing display
    display_img = Display("display_image")
    display_img2 = Display("display_image2")
    
    
    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # escala de grises
        grey_image = greyscale_cv2(image)
        
        # blurring
        img_blur = cv2.GaussianBlur(grey_image, (3,3), 0, 0)
        
        # detecto con Canny edge 
        img_canny = cv2.Canny(img_blur, 50, 150)
       
        
        # Tamaño de la imagen
        height = 64
        width = 128

        # Definir los vértices del ROI
        vertices = np.array([[(20, 45), (0, height), (width, height), (108, 45)]], dtype=np.int32)

        # Crear una imagen negra del mismo tamaño que la imagen original
        img_roi = np.zeros_like(grey_image)

        # Rellenar el ROI con blanco
        cv2.fillPoly(img_roi, vertices, 255)

        # Aplicar la máscara del ROI a la imagen Canny
        img_mask = cv2.bitwise_and(img_canny, img_roi)
        
        # parametros de Hugh
        rho = 1.3
        theta = (np.pi*2.1) / 180
        threshold = 5
        min_line_length = 1
        max_line_gap = 10
        
        # Hough line detection
        lines = cv2.HoughLinesP(img_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
        
        alpha = 1
        beta = 1
        gamma= 1
        
        img_lane_lines = cv2.addWeighted(img_rgb, alpha, img_lines, beta, gamma)
        
        # Process detected lines
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 30)
        
            # ángulo promedio de las líneas detectadas
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            avg_angle = np.mean(angles)
            # Print avrage angle
            print("Average Angle: {:.2f}".format(avg_angle))
            
            steering_angle = avg_angle * 0.02  # Adjust the scaling factor as needed
            set_steering_angle(steering_angle)
        else:
            set_steering_angle(0.0)
        
        # Display the processed image
        display_image(display_img, img_mask)
        display_image(display_img2, img_lane_lines)
       
        
        # Update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()