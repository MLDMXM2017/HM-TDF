import os
from PIL import Image

img_dir = './Tongue-FLD/Tongue_Images/'
output_folder_father ='./Tongue-FLD/Tongue_Images_rotated/' 
angle_hub = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] # rotate_angle
pad_value = (0, 0, 0) 
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

for angle in angle_hub:
    output_folder = os.path.join(output_folder_father, str(angle))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(img_dir, filename)
            output_path = os.path.join(output_folder, filename)
            with Image.open(input_path) as img:
                rotated_img = img.rotate(
                    angle=-angle,             # Rotate clockwise, use negative value
                    resample=Image.NEAREST,   # Interpolation method, options are NEAREST, BILINEAR, BICUBIC
                    expand=True,              # Expand image size to fit the rotated dimensions
                    fillcolor=pad_value       # Fill color
                )
                rotated_img.save(output_path)
    print(angle, 'done!')

