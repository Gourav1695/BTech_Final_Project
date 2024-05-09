import cv2
import os

def preProcessing(myImage, output_folder, img_count):
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)
    horizontal_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = myImage.copy()
    word_images = []  # List to store segmented word images
    for cnt in horizontal_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out contours based on bounding box dimensions
        if 50 < w < 299 and h < 199:
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 0)
            word_img = myImage[y:y+h, x:x+w]  # Extract the segmented word region
            word_images.append(word_img)
    save_images(word_images, output_folder, img_count)
    return word_images

def save_images(images, output_folder, img_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, img in enumerate(images):
        h, w, _ = img.shape
        img_no = img_count * 1000 + i + 1
        cv2.imwrite(os.path.join(output_folder, f'{img_no}-{h}-{w}.jpg'), img)
