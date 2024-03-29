import tifffile as tiff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the mask and transpose it
mask = tiff.imread('classificationROItif.tif')
mask = np.transpose(mask)
print("The mask shape is: ", mask.shape)

#import the image and transpose it
image = tiff.imread('Mosaic_raw1.tif')
image = np.transpose(image)
print("The image shape is: ", image.shape)

#Take the unique values from the mask and have an array that is only the 3 RGB values
arr_mask_rgb = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)

#Create an array that contains all zeroes that will act as the mask
arr_mask_copy = np.zeros((1800, 1500), dtype=int)

#Loop through and when match the imported mask RGB values with the array mask RGB values and change them to 1 or 2
for i in range(0, mask.shape[0]):
    for j in range (0, mask.shape[1]):
        if (np.array_equal(mask[i][j], arr_mask_rgb[1])):
            arr_mask_copy[i][j] = 1
        if (np.array_equal(mask[i][j], arr_mask_rgb[2])):
            arr_mask_copy[i][j] = 2


df_mask = pd.DataFrame(arr_mask_copy)
mask_csv = df_mask.to_csv('mask.csv')


#Create an empty dataframe
df_pixel = pd.DataFrame()

blue_arr = []
green_arr = []

#Loop through each band and find when the arr_mask_copy is equal to 1 or 2 and append to the corresponding array.
#This then stacks those two arrays and adds them to the dataframe as a single column
for i in range(0, image.shape[2]):
    blue_arr = []
    green_arr = []
    for j in range (0, image.shape[0]):
        for k in range(0, image.shape[1]):
            if(arr_mask_copy[j][k] == 1):
                blue_arr = np.append(blue_arr, image[j][k][i])
            if(arr_mask_copy[j][k] == 2):
                green_arr = np.append(green_arr, image[j][k][i])
    
    blue_green = np.hstack((blue_arr, green_arr))
    df_pixel[i] = blue_green.tolist()

display(df_pixel)
image_csv = df_pixel.to_csv('image_raw.csv')