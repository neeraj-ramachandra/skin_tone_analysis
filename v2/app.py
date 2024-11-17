import random
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_palette_colors(image):
    pixels = image.reshape(-1, 3)
    # Use KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_

    # Convert colors to int
    colors = colors.astype(int)
    colors = colors[np.any(colors != [0, 0, 0], axis=1)]
    colors = np.array(colors, dtype=np.uint8)
    return colors

def plot_palette_colors(skin_image, action='display'):
    colors = get_palette_colors(cv2.cvtColor(skin_image, cv2.COLOR_RGB2BGR))
    block_size = 50
    palette_width = len(colors) * block_size
    palette = np.zeros((block_size, palette_width, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        start_x = i * 50
        end_x = start_x + 50
        palette[:, start_x:end_x] = color

    palette_rgb = cv2.cvtColor(palette, cv2.COLOR_BGR2RGB)

    # Display the color palette
    plt.figure(num='Skin Tone Palette')
    plt.imshow(palette_rgb)
    plt.axis('off')
    for i, name in enumerate(get_palette_colors(skin_image)):
        plt.text(i * block_size + block_size // 2, block_size // 2, f"RGB{tuple(map(int, name))}",
                color='white', fontsize=8, ha='center', va='center')

    if action == 'display':
        plt.show()
    elif action == 'save':
        # Save the color palette
        filepath = "palette/" + str(random.randint(100000, 999999)) + '_palette.png'
        plt.savefig(f'{filepath}', format='png', dpi=300, bbox_inches='tight')
        print(f"Color palette saved to '{filepath}'")

def detect_skin_tone(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face and crop it
    for (x, y, w, h) in faces:
        # Crop the face
        face = image[y:y+h, x:x+w]


    image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define skin tone range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin tone
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply the mask to the image
    skin_image = cv2.bitwise_and(image, image, mask=skin_mask)

    # Convert the result to grayscale to simplify further processing
    gray_skin_image = cv2.cvtColor(skin_image, cv2.COLOR_RGB2GRAY)
    
    print ("Select Display Options: \n 1. OpenCv \n 2. Metplotlib \n 3. Color Palette \n 4. Save Color Palette")
    option = int(input("Enter your option: "))

    if option == 1:
        # Display the result
        cv2.imshow('Skin Tone Detection', cv2.cvtColor(skin_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif option == 2:
        # Display the result
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Skin Mask')
        gray_skin_image_rgb = cv2.cvtColor(gray_skin_image, cv2.COLOR_BGR2RGB)
        plt.imshow(gray_skin_image_rgb)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Skin Tone Detection')
        plt.imshow(skin_image)
        plt.axis('off')
        plt.show()
    elif option == 3:
        plot_palette_colors(skin_image, action='display')
    elif option == 4:
        plot_palette_colors(skin_image, action='save')

# Example usage
detect_skin_tone('images/file1.jpg')