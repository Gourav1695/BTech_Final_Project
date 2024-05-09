import cv2
import glob
from pre_processing import preProcessing

def main():
    jpgImages = glob.glob("jpgImgs/*.JPG")
    img_count = 0  # Initialize image count

    for jpg in jpgImages:
        img_count += 1  # Increment image count for each new image
        print(jpg)
        image = cv2.imread(jpg)
        output_folder = f'{jpg[:-4]}_words'
        preProcessing(image, output_folder, img_count)

if __name__ == "__main__":
    main()
