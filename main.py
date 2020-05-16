"""This programme categorises a photo of soil as either dry, damp, or wet"""

import cv2
import numpy as np
import matplotlib.pyplot as plot

"""General variables"""
MASK_COLOUR = (0.0, 0.5, 1.0)  # In BGR format
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
THRESHOLD1 = 300
THRESHOLD2 = 900
IMAGE_LIST = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg", "12.jpg", "13.jpg"]
B_VALS = []
G_VALS = []
R_VALS = []

"""Thresholds which were calculated by graphing RGB values of the soil from 13 different samples"""
WET_THRESHOLD = 25
DAMP_THRESHOLD = 40


"""Balances the colours by stretching the RGB values across the full spectrum"""
def colour_balance(img, percent):
    out_channels = []
    channels = cv2.split(img)
    totalstop = channels[0].shape[0] * channels[0].shape[1] * percent / 200.0
    for channel in channels:
        bc = np.bincount(channel.ravel(), minlength=256)
        lv = np.searchsorted(np.cumsum(bc), totalstop)
        hv = 255-np.searchsorted(np.cumsum(bc[::-1]), totalstop)
        out_channels.append(cv2.LUT(channel, np.array(tuple(0 if i < lv else 255 if i > hv else round((i-lv)/(hv-lv)*255) for i in np.arange(0, 256)), dtype="uint8")))
    return cv2.merge(out_channels)


"""Finds contours of objects in image"""
def contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (19, 19), 0)  # Gaussian blur to reduce noise in the image.
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)  # Use adaptive thresholding to "binarise" the image.
    # Perform some morphological operations to help distinguish some of the features in the image.
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Fit ellipse around soil
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:  # Set a lower bound on the ellipse area.
            continue

        if len(contour) < 5:  # The fitEllipse function requires at least five points on the contour to function.
            continue
        desired_contour = contour

    return desired_contour, closing


"""Creates a mask using thresholding so the background of the soil is able to be disregarded"""
def threshold_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(grayscale_image, 160, 255, cv2.THRESH_BINARY_INV)
    dilate_kernel = np.ones((3, 3), np.uint8)
    erode_kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, dilate_kernel)
    mask = cv2.erode(mask, erode_kernel, iterations=6)
    return mask


"""Graphs each colour channel"""
def graph_BGR_vals(totaled_colour_list):
    x = range(13)
    plot.plot(x, B_VALS, label="B")
    plot.plot(x, G_VALS, label="G")
    plot.plot(x, R_VALS, label="R")
    plot.plot(x, totaled_colour_list, label="sum of all")
    plot.legend()
    plot.title("BGR values")
    plot.show()


"""Combines all RGB values and finds the average which is used for determining soil moisture"""
def get_total_colour():
    summed_colour_list = []
    for number in range(13):
        total = (B_VALS[number] + G_VALS[number] + R_VALS[number]) / 3
        summed_colour_list.append(total)
    return summed_colour_list


"""Function to discretise moisture levels and puts image into one of the three categories"""
def categorise_moisture(summed_colour_list):
    moisture_dict = {}
    for image_number in range(len(summed_colour_list)):
        if summed_colour_list[image_number] < WET_THRESHOLD:
            moisture_dict[IMAGE_LIST[image_number]] = "wet"
        elif summed_colour_list[image_number] < DAMP_THRESHOLD:
            moisture_dict[IMAGE_LIST[image_number]] = "damp"
        else:
            moisture_dict[IMAGE_LIST[image_number]] = "dry"
    return moisture_dict


"""Main function showing running order of the code"""
def main():
    for image_name in IMAGE_LIST:
        image = cv2.imread(image_name)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        colour_balanced = colour_balance(image, 1)
        mask = threshold_image(colour_balanced)
        final_image = cv2.bitwise_and(colour_balanced, colour_balanced, mask=mask)
        average_BGR_val = cv2.mean(final_image, mask=mask)
        B_VALS.append(average_BGR_val[0])
        G_VALS.append(average_BGR_val[1])
        R_VALS.append(average_BGR_val[2])
    summed_colour_list = get_total_colour()
    moisture_dict = categorise_moisture(summed_colour_list)
    # graph_BGR_vals(summed_colour_list)
    for x in moisture_dict:
        print("image", x)
        print(moisture_dict[x])
    cv2.waitKey(0)


main()
