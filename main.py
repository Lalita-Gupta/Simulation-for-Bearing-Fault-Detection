# Importing Packages
# **************************************************************************************************************************************************************
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import streamlit as st 
import csv
# import pywt

# All Feature Matrics Functions
# **************************************************************************************************************************************************************
def mean(image):
    mask_size = 2
    avg_total = []

    for c in range (0,image.shape[1]-mask_size+1):
        # avg= []
        total = 0
        count = 0
        for r in range (0,image.shape[0]-mask_size+1):
            s = np.sum(image[r:r+mask_size,c:c+mask_size])
            d = mask_size * mask_size
            temp = s/d
            total = total + temp
            count = count + 1
            # avg.append(temp)
        total = total / count
        # avg.append(total)
        avg_total.append(total)
        # with open("mean.csv", 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(avg)

    st.subheader("Mean Line Chart")
    avg_total_chart = pd.DataFrame(avg_total)
    st.line_chart(avg_total_chart)
    return avg_total

def std(image):
    mask_size = 2
    std_total = []

    for c in range (0,image.shape[1]-mask_size+1):
        # std = []
        total = 0
        count = 0
        for r in range (0,image.shape[0]-mask_size+1):
            s = np.sum(image[r:r+mask_size,c:c+mask_size])
            d = mask_size * mask_size
            avg = s/d
            temp2 = 0
            for i in range(0,mask_size):
                for j in range(0,mask_size): 
                    # pixel_b, pixel_g, pixel_r = image[r+i][c+j]
                    # t = (pixel_b + pixel_g + pixel_r) / 3
                    t = image[r+i][c+j]
                    temp2 = temp2 + (t-avg)**2
            temp = np.sqrt(temp2/d)
            total = total + temp
            count = count + 1
            # std.append(temp)
        total = total / count
        std_total.append(total)
        # std.append(total)
        # with open("std.csv", 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(std)

    st.subheader("Standard Deviation Line Chart")

    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
        std_total_chart = pd.DataFrame(std_total,columns=["Blue", "Green", "Red"])
    else:
        std_total_chart = pd.DataFrame(std_total)

    st.line_chart(std_total_chart)
    return std_total

def var(image):
    mask_size = 2
    var_total = []

    for c in range (0,image.shape[1]-mask_size+1):
        # var = []
        total = 0
        count = 0
        for r in range (0,image.shape[0]-mask_size+1):
            s = np.sum(image[r:r+mask_size,c:c+mask_size])
            d = mask_size * mask_size
            avg = s/d
            temp2 = 0
            for i in range(0,mask_size):
                for j in range(0,mask_size): 
                    #pixel_b, pixel_g, pixel_r = image[r+i][c+j]
                    #t = (pixel_b + pixel_g + pixel_r) / 3
                    t = image[r+i][c+j]
                    temp2 = temp2 + (t-avg)**2
            temp = temp2/d
            total = total + temp
            count = count + 1
            # var.append(temp)
        total = total / count
        var_total.append(total)
        # var.append(total)
        # with open("var.csv", 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(var)
    
    st.subheader("Variance Line Chart")

    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
        var_total_chart = pd.DataFrame(var_total,columns=["Blue", "Green", "Red"])
    else:
        var_total_chart = pd.DataFrame(var_total)

    st.line_chart(var_total_chart)
    return var_total

def rms(image):
    mask_size = 2
    rms_total = []

    for c in range (0,image.shape[1]-mask_size+1):
        # rms = []
        total = 0
        count = 0
        for r in range (0,image.shape[0]-mask_size+1):
            s = np.sum(image[r:r+mask_size,c:c+mask_size])
            d = mask_size * mask_size
            avg = s/d
            temp2 = 0
            for i in range(0,mask_size):
                for j in range(0,mask_size): 
                    #pixel_b, pixel_g, pixel_r = image[r+i][c+j]
                    #t = (pixel_b + pixel_g + pixel_r) / 3
                    t = image[r+i][c+j]
                    temp2 = temp2 + t*t
            temp = np.sqrt(temp2/d)
            total = total + temp
            count = count + 1
            # rms.append(temp)
        total = total / count
        rms_total.append(total)
        # rms.append(total)
        # with open("rms.csv", 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(rms)

    st.subheader("Root Mean Square Line Chart")
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
        rms_total_chart = pd.DataFrame(rms_total,columns=["Blue", "Green", "Red"])
    else:
        rms_total_chart = pd.DataFrame(rms_total)

    st.line_chart(rms_total_chart)
    return rms_total

def histogram(image):
        st.subheader("Histogram")
        hist = cv2.calcHist([image],[2],None,[256],[0,500])
        st.line_chart(hist)

def table(image):
    # dictionary of lists 
    dict = {"Mean": avg_total, "STD": std_total, "Var": var_total, "RMS": rms_total} # "MSE": mse_total
    df = pd.DataFrame(dict)
    st.subheader("Table")
    st.dataframe(df)
    if choice4 == "HUE Coloration" or choice2 == "Canny Edge Detection" or choice2 == "Otsu Edge Detection" or (choice6 == "Gamma Transformation" and choice4 == "Gray Coloration"):
        st.subheader("Combined Graph")
        st.line_chart(df)

    d = image.shape[1] - 1
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
        st.subheader("Total Mean:")
        st.write(sum(avg_total)/d)
        st.subheader("Total STD:")
        st.write(sum(sum(std_total))/d)
        st.subheader("Total Variance:")
        st.write(sum(sum(var_total))/d)
        st.subheader("Total RMS:")
        st.write(sum(sum(rms_total))/d)
    else:
        st.subheader("Total Mean:")
        st.write(sum(avg_total)/d)
        st.subheader("Total STD:")
        st.write(sum(std_total)/d)
        st.subheader("Total Variance:")
        st.write(sum(var_total)/d)
        st.subheader("Total RMS:")
        st.write(sum(rms_total)/d)

# All Edge Detection Functions
# **************************************************************************************************************************************************************
def canny(image):
    edges = cv2.Canny(image,100,200)
    return edges

def otsu(image):
    edges = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return edges

def prewitt(image):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    edges = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
    return edges

def robert(image):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    img_robertx = cv2.filter2D(image, -1, kernelx)
    img_roberty = cv2.filter2D(image, -1, kernely)
    edges = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)
    return edges

# Streamlit Simulation
# **************************************************************************************************************************************************************

# Title
st.title("Simulation for Bearing Fault Detection")

# Sidebar
with st.sidebar:
    choice_img = st.selectbox("Image Set", ["Select one", "Set1 - 30% Phase1, 50% Phase1", "Set2 - 30% Phase2, 50% Phase1", "Set3 - 30% Phase3, 50% Phase1", "Set4 - 30% Phase1, 50% Phase2", "Set5 - 30% Phase2, 50% Phase2", "Set6 - 30% Phase3, 50% Phase2"])

    if choice_img != "Select one":
        on1 = st.toggle('Original Image Histogram')
        choice6 = st.selectbox("Image Transformation", ["Select one", "No Transformation", "Gamma Transformation", "Log Transformation", "Inverse Log Transformation"])

        if choice6 != "Select one":
            on2 = st.toggle('Transformed Image Histogram')
            choice4 = st.selectbox("Image Coloration", ["Select one", "No Coloration Image", "Gray Coloration", "HUE Coloration", "Pseudo Coloration"])

            if choice4 != "Select one":

                if choice4 == "Pseudo Coloration":
                    choice5 = st.selectbox("Types of Pseudo Coloration", ["Select one", "Spring", "Hot", "Cool", "Rainbow", "HSV", "JET"])
                    on3 = st.toggle('Coloration Image Histogram')

                    if choice5 != "Select one":
                        
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                        if choice2 != 'Select one':
                            choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])
                            on4 = st.toggle('Edge Detection Histogram')

                else:
                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                        on3 = st.toggle('Coloration Image Histogram')

                    if choice4 == "HUE Coloration":
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                        if choice2 != 'Select one':
                            choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])
                    else:
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                        if choice2 != 'Select one':
                            choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"]) 
                            if choice4 != "Gray Coloration":
                                on4 = st.toggle('Edge Detection Histogram')

        
# Creating of Columns
col1, col2, col3 = st.columns([1,1,1])

# All Options 
# **************************************************************************************************************************************************************

if choice_img != "Select one":

    st.header('', divider='rainbow')

    # Image sets
    if choice_img == "Set1 - 30% Phase1, 50% Phase1":
        with col1: 
            img1 = cv2.imread("Noload/004.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("Healthy - No Load")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            if on1:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A30/137.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Phase 1")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            if on1:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A50/256.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Phase 1")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            if on1:
                histogram(img3)

    if choice_img == "Set2 - 30% Phase2, 50% Phase1":
        with col1:
            img1 = cv2.imread("Noload/005.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("Healthy - No Load")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            if on1:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C30/177.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Phase 2")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            if on1:
                histogram(img2)
        
        with col3:
            img3 = cv2.imread("A50/258.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Phase 1")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            if on1:
                histogram(img3)

    if choice_img == "Set3 - 30% Phase3, 50% Phase1":
        with col1:
            img1 = cv2.imread("Noload/006.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("Healthy - No Load")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            if on1:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C&B30/217.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Phase 3")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            if on1:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A50/259.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Phase 1")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            if on1:
                histogram(img3)

    if choice_img == "Set4 - 30% Phase1, 50% Phase2":
        with col1:
            img1 = cv2.imread("Noload/007.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("Healthy - No Load")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            if on1:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A30/134.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Phase 1")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            if on1:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A&B50/292.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Phase 2")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            if on1:
                histogram(img3)

    if choice_img == "Set5 - 30% Phase2, 50% Phase2":
        with col1:
            img1 = cv2.imread("Noload/008.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("Healthy - No Load")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            if on1:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C30/180.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Phase 2")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            if on1:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A&B50/293.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Phase 2")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            if on1:
                histogram(img3)

    if choice_img == "Set6 - 30% Phase3, 50% Phase2":
        with col1:
            img1 = cv2.imread("Noload/008.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("Healthy - No Load")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            if on1:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C&B30/215.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Phase 3")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            if on1:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A&B50/293.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Phase 2")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            if on1:
                histogram(img3)

    # Image Transformation and Coloration
    # **************************************************************************************************************************************************************
    if choice6 != "Select one":

        if choice6 == "No Transformation":
            with col1:

                # Display the hue image
                st.subheader("No Transformation - No Load Image")
                st.image(img1, caption = "No Transformation NO LOAD Image")
                st.write("Image dimensions:", img1.shape)
                apply_image1 = img1

            with col2:

                # Display the hue image
                st.subheader("No Transformation - 30% Load Image")
                st.image(img2, caption = "No Transformation 30% LOAD Image")
                st.write("Image dimensions:", img2.shape)
                apply_image2 = img2

            with col3:

                # Display the hue image
                st.subheader("No Transformation - 50% Load Image")
                st.image(img3, caption = "No Transformation 50% LOAD Image")
                st.write("Image dimensions:", img3.shape)
                apply_image3 = img3

        if choice6 == "Gamma Transformation":

            with col1:

                # Define the gamma value (adjust as needed)
                gamma = 0.1

                # Perform gamma correction
                gamma_corrected = np.power(img1 / 255.0, gamma) * 255.0
                gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

                st.subheader("Gamma Transformation - No Load Image")
                st.image(gamma_corrected, caption = "Gamma Transformation - NO LOAD Image")
                st.write("Image dimensions:", gamma_corrected.shape)
                img1 = gamma_corrected

            with col2:

                # Define the gamma value (adjust as needed)
                gamma = 0.1

                # Perform gamma correction
                gamma_corrected = np.power(img2 / 255.0, gamma) * 255.0
                gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

                st.subheader("Gamma Transformation - 30% Load Image")
                st.image(gamma_corrected, caption = "Gamma Transformation - 30% LOAD Image")
                st.write("Image dimensions:", gamma_corrected.shape)
                img2 = gamma_corrected

            with col3:

                # Define the gamma value (adjust as needed)
                gamma = 0.1

                # Perform gamma correction
                gamma_corrected = np.power(img3 / 255.0, gamma) * 255.0
                gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

                st.subheader("Gamma Transformation - 50% Load Image")
                st.image(gamma_corrected, caption = "Gamma Transformation - 50% LOAD Image")
                st.write("Image dimensions:", gamma_corrected.shape)
                img3 = gamma_corrected

        if choice6 == "Log Transformation":

            with col1:
                # Split the image into its color channels
                b, g, r = cv2.split(img1)

                # Apply log transformation to each color channel
                c = 1  # Constant value to avoid log(0)
                log_transformed_b = c * np.log1p(b.astype(np.float32))
                log_transformed_g = c * np.log1p(g.astype(np.float32))
                log_transformed_r = c * np.log1p(r.astype(np.float32))

                # Scale the values to 0-255 range
                log_transformed_b = cv2.normalize(log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                log_transformed_g = cv2.normalize(log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                log_transformed_r = cv2.normalize(log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Merge the log-transformed color channels back into an image
                log_transformed_image = cv2.merge((log_transformed_b, log_transformed_g, log_transformed_r))

                st.subheader("Log Transformation - No Load Image")
                st.image(log_transformed_image, caption = "Log Transformation -- NO LOAD Image")
                st.write("Image dimensions:", log_transformed_image.shape)
                img1 = log_transformed_image

            with col2:
                # Split the image into its color channels
                b, g, r = cv2.split(img2)

                # Apply log transformation to each color channel
                c = 1  # Constant value to avoid log(0)
                log_transformed_b = c * np.log1p(b.astype(np.float32))
                log_transformed_g = c * np.log1p(g.astype(np.float32))
                log_transformed_r = c * np.log1p(r.astype(np.float32))

                # Scale the values to 0-255 range
                log_transformed_b = cv2.normalize(log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                log_transformed_g = cv2.normalize(log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                log_transformed_r = cv2.normalize(log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Merge the log-transformed color channels back into an image
                log_transformed_image = cv2.merge((log_transformed_b, log_transformed_g, log_transformed_r))

                st.subheader("Log Transformation - 30% Load Image")
                st.image(log_transformed_image, caption = "Log Transformation - 30% LOAD Image")
                st.write("Image dimensions:", log_transformed_image.shape)
                img2 = log_transformed_image

            with col3:
                # Split the image into its color channels
                b, g, r = cv2.split(img3)

                # Apply log transformation to each color channel
                c = 1  # Constant value to avoid log(0)
                log_transformed_b = c * np.log1p(b.astype(np.float32))
                log_transformed_g = c * np.log1p(g.astype(np.float32))
                log_transformed_r = c * np.log1p(r.astype(np.float32))

                # Scale the values to 0-255 range
                log_transformed_b = cv2.normalize(log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                log_transformed_g = cv2.normalize(log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                log_transformed_r = cv2.normalize(log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Merge the log-transformed color channels back into an image
                log_transformed_image = cv2.merge((log_transformed_b, log_transformed_g, log_transformed_r))

                st.subheader("Log Transformation - 50% Load Image")
                st.image(log_transformed_image, caption = "Log Transformation - 50% LOAD Image")
                st.write("Image dimensions:", log_transformed_image.shape)
                img3 = log_transformed_image

        if choice6 == "Inverse Log Transformation":
            
            with col1:
                # Split the original image into its color channels
                b, g, r = cv2.split(img1)

                # Apply log transformation to each color channel
                c = 1  # Constant value to avoid log(0)
                log_transformed_b = c * np.log1p(b.astype(np.float32))
                log_transformed_g = c * np.log1p(g.astype(np.float32))
                log_transformed_r = c * np.log1p(r.astype(np.float32))

                # Apply inverse log transformation to each color channel
                inv_log_transformed_b = np.expm1(log_transformed_b)
                inv_log_transformed_g = np.expm1(log_transformed_g)
                inv_log_transformed_r = np.expm1(log_transformed_r)

                # Scale the values to 0-255 range
                inv_log_transformed_b = cv2.normalize(inv_log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                inv_log_transformed_g = cv2.normalize(inv_log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                inv_log_transformed_r = cv2.normalize(inv_log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Merge the inverse log-transformed color channels back into an image
                inv_log_transformed_image = cv2.merge((inv_log_transformed_b, inv_log_transformed_g, inv_log_transformed_r))

                # Display the image
                st.subheader("Inverse Log Transformation - No Load Image")
                st.image(inv_log_transformed_image, caption = "Inverse Log Transformation - NO LOAD Image")
                st.write("Image dimensions:", inv_log_transformed_image.shape)
                img1 = inv_log_transformed_image

            with col2:
                # Split the original image into its color channels
                b, g, r = cv2.split(img2)

                # Apply log transformation to each color channel
                c = 1  # Constant value to avoid log(0)
                log_transformed_b = c * np.log1p(b.astype(np.float32))
                log_transformed_g = c * np.log1p(g.astype(np.float32))
                log_transformed_r = c * np.log1p(r.astype(np.float32))

                # Apply inverse log transformation to each color channel
                inv_log_transformed_b = np.expm1(log_transformed_b)
                inv_log_transformed_g = np.expm1(log_transformed_g)
                inv_log_transformed_r = np.expm1(log_transformed_r)

                # Scale the values to 0-255 range
                inv_log_transformed_b = cv2.normalize(inv_log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                inv_log_transformed_g = cv2.normalize(inv_log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                inv_log_transformed_r = cv2.normalize(inv_log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Merge the inverse log-transformed color channels back into an image
                inv_log_transformed_image = cv2.merge((inv_log_transformed_b, inv_log_transformed_g, inv_log_transformed_r))

                # Display the image
                st.subheader("Inverse Log Transformation - 30% Load Image")
                st.image(inv_log_transformed_image, caption = "Inverse Log Transformation - 30% LOAD Image")
                st.write("Image dimensions:", inv_log_transformed_image.shape)
                img2 = inv_log_transformed_image

            with col3:
                # Split the original image into its color channels
                b, g, r = cv2.split(img3)

                # Apply log transformation to each color channel
                c = 1  # Constant value to avoid log(0)
                log_transformed_b = c * np.log1p(b.astype(np.float32))
                log_transformed_g = c * np.log1p(g.astype(np.float32))
                log_transformed_r = c * np.log1p(r.astype(np.float32))

                # Apply inverse log transformation to each color channel
                inv_log_transformed_b = np.expm1(log_transformed_b)
                inv_log_transformed_g = np.expm1(log_transformed_g)
                inv_log_transformed_r = np.expm1(log_transformed_r)

                # Scale the values to 0-255 range
                inv_log_transformed_b = cv2.normalize(inv_log_transformed_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                inv_log_transformed_g = cv2.normalize(inv_log_transformed_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                inv_log_transformed_r = cv2.normalize(inv_log_transformed_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Merge the inverse log-transformed color channels back into an image
                inv_log_transformed_image = cv2.merge((inv_log_transformed_b, inv_log_transformed_g, inv_log_transformed_r))

                # Display the image
                st.subheader("Inverse Log Transformation - 50% Load Image")
                st.image(inv_log_transformed_image, caption = "Inverse Log Transformation - 50% LOAD Image")
                st.write("Image dimensions:", inv_log_transformed_image.shape)
                img3 = inv_log_transformed_image

        if on2:
                with col1:
                    histogram(img1)
                with col2:
                    histogram(img2)
                with col3:
                    histogram(img3)

        if choice4 == "No Coloration Image":

            with col1:

                st.subheader("No Coloration - No Load Image")
                st.image(img1, caption = "No Coloration - NO LOAD Image")
                st.write("Image dimensions:", img1.shape)
                apply_image1 = img1

            with col2:

                st.subheader("No Coloration - 30% Load Image")
                st.image(img2, caption = "No Coloration - 30% LOAD Image")
                st.write("Image dimensions:", img2.shape)
                apply_image2 = img2

            with col3:

                st.subheader("No Coloration - 50% Load Image")
                st.image(img3, caption = "No Coloration - 50% LOAD Image")
                st.write("Image dimensions:", img3.shape)
                apply_image3 = img3

        if choice4 == "Gray Coloration":

            with col1:

                gray_image1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

                st.subheader("Gray Coloration - No Load Image")
                st.image(gray_image1, caption = "Gray Coloration - NO LOAD Image")
                st.write("Image dimensions:", gray_image1.shape)
                apply_image1 = gray_image1

            with col2:

                gray_image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

                st.subheader("Gray Coloration - 30% Load Image")
                st.image(gray_image2, caption = "Gray Coloration - 30% LOAD Image")
                st.write("Image dimensions:", gray_image2.shape)
                apply_image2 = gray_image2

            with col3:

                gray_image3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

                st.subheader("Gray Coloration - 50% Load Image")
                st.image(gray_image3, caption = "Gray Coloration - 50% LOAD Image")
                st.write("Image dimensions:", gray_image3.shape)
                apply_image3 = gray_image3

        if choice4 == "HUE Coloration":
                
            with col1:

                # Convert BGR image to HSV
                hsv_image1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

                # Extract the hue channel
                hue_image1 = hsv_image1[:, :, 0]  # Hue channel is the first channel in HSV

                # Display the hue image
                st.subheader("HUE Coloration - No Load Image")
                st.image(hue_image1, caption = "HUE NO LOAD Image", channels='HSV', use_column_width=True)
                st.write("Image dimensions:", hue_image1.shape)
                apply_image1 = hue_image1

            with col2:

                # Convert BGR image to HSV
                hsv_image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

                # Extract the hue channel
                hue_image2 = hsv_image2[:, :, 0]  # Hue channel is the first channel in HSV

                # Display the hue image
                st.subheader("HUE Coloration - 30% Load Image")
                st.image(hue_image2, caption = "HUE 30% LOAD Image", channels='HSV', use_column_width=True)
                st.write("Image dimensions:", hue_image2.shape)
                apply_image2 = hue_image2

            with col3:
            
                # Convert BGR image to HSV
                hsv_image3 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV)

                # Extract the hue channel
                hue_image3 = hsv_image3[:, :, 0]  # Hue channel is the first channel in HSV

                # Display the hue image
                st.subheader("HUE Coloration - 50% Load Image")
                st.image(hue_image3, caption = "HUE 50% LOAD Image", channels='HSV', use_column_width=True)
                st.write("Image dimensions:", hue_image3.shape)
                apply_image3 = hue_image3

        if choice4 == "Pseudo Coloration":

            if choice5 == "Spring":

                with col1:

                    pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_SPRING)

                    st.subheader("Pesudo Spring Coloration Image")
                    st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                    st.write("Image dimensions:", pesudo_image1.shape)
                    apply_image1 = pesudo_image1

                with col2:

                    pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_SPRING)

                    st.subheader("Pesudo Spring Coloration Image")
                    st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                    st.write("Image dimensions:", pesudo_image2.shape)
                    apply_image2 = pesudo_image2

                with col3:
                
                    pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_SPRING)

                    st.subheader("Pesudo Spring Coloration Image")
                    st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                    st.write("Image dimensions:", pesudo_image3.shape)
                    apply_image3 = pesudo_image3

            if choice5 == "Hot":

                with col1:

                    pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_HOT)

                    st.subheader("Pesudo Hot Coloration Image")
                    st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                    st.write("Image dimensions:", pesudo_image1.shape)
                    apply_image1 = pesudo_image1

                with col2:

                    pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_HOT)

                    st.subheader("Pesudo Hot Coloration Image")
                    st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                    st.write("Image dimensions:", pesudo_image2.shape)
                    apply_image2 = pesudo_image2

                with col3:
                
                    pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_HOT)

                    st.subheader("Pesudo Hot Coloration Image")
                    st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                    st.write("Image dimensions:", pesudo_image3.shape)
                    apply_image3 = pesudo_image3

            if choice5 == "Cool":

                with col1:

                    pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_COOL)

                    st.subheader("Pesudo Cool Coloration Image")
                    st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                    st.write("Image dimensions:", pesudo_image1.shape)
                    apply_image1 = pesudo_image1

                with col2:

                    pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_COOL)

                    st.subheader("Pesudo Cool Coloration Image")
                    st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                    st.write("Image dimensions:", pesudo_image2.shape)
                    apply_image2 = pesudo_image2

                with col3:
                
                    pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_COOL)

                    st.subheader("Pesudo Cool Coloration Image")
                    st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                    st.write("Image dimensions:", pesudo_image3.shape)
                    apply_image3 = pesudo_image3

            if choice5 == "Rainbow":

                with col1:

                    pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_RAINBOW)

                    st.subheader("Pesudo Rainbow Coloration Image")
                    st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                    st.write("Image dimensions:", pesudo_image1.shape)
                    apply_image1 = pesudo_image1

                with col2:

                    pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_RAINBOW)

                    st.subheader("Pesudo Rainbow Coloration Image")
                    st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                    st.write("Image dimensions:", pesudo_image2.shape)
                    apply_image2 = pesudo_image2

                with col3:
                
                    pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_RAINBOW)

                    st.subheader("Pesudo Rainbow Coloration Image")
                    st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                    st.write("Image dimensions:", pesudo_image3.shape)
                    apply_image3 = pesudo_image3

            if choice5 == "HSV":
                
                with col1:

                    pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_HSV)

                    st.subheader("Pesudo HSV Coloration Image")
                    st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                    st.write("Image dimensions:", pesudo_image1.shape)
                    apply_image1 = pesudo_image1

                with col2:

                    pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_HSV)

                    st.subheader("Pesudo HSV Coloration Image")
                    st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                    st.write("Image dimensions:", pesudo_image2.shape)
                    apply_image2 = pesudo_image2

                with col3:
                
                    pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_HSV)

                    st.subheader("Pesudo HSV Coloration Image")
                    st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                    st.write("Image dimensions:", pesudo_image3.shape)
                    apply_image3 = pesudo_image3

            if choice5 == "JET":

                with col1:

                    pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_JET)

                    st.subheader("Pesudo JET Coloration Image")
                    st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                    st.write("Image dimensions:", pesudo_image1.shape)
                    apply_image1 = pesudo_image1

                with col2:

                    pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)

                    st.subheader("Pesudo JET Coloration Image")
                    st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                    st.write("Image dimensions:", pesudo_image2.shape)
                    apply_image2 = pesudo_image2

                with col3:
                
                    pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_JET)

                    st.subheader("Pesudo JET Coloration Image")
                    st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                    st.write("Image dimensions:", pesudo_image3.shape)
                    apply_image3 = pesudo_image3

        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice4 != "Select one":

            if on3:
                    if choice4 == 'Pseudo Coloration' and choice5 == "Select one":
                        pass
                    else:
                        with col1:
                            histogram(apply_image1)
                        with col2:
                            histogram(apply_image2)
                        with col3:
                            histogram(apply_image3)

    # After Image Transformation and Coloration
    # **************************************************************************************************************************************************************
    if choice_img != "Select one" and choice6 != "Select one" and choice4 != "Select one":

        if choice4 == "Pseudo Coloration" and choice5 == "Select one":
            pass 

        else:

            # Edge Detection 
            # **************************************************************************************************************************************************************

            # for Canny
            if choice2 == "Canny Edge Detection":

                with col1:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image1

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image1, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image1,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image1,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image1, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image1,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image1, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)

                        edges1 = canny(image_result)

                        st.subheader("Canny Edge Detection Image")
                        st.image(edges1, caption = "CANNY EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table(edges1)
                    
                with col2:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image2

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image2, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image2,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image2,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image2, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image2,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image2, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)

                        edges2 = canny(image_result)

                        st.subheader("Canny Edge Detection Image")
                        st.image(edges2, caption = "CANNY EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table(edges2)


                with col3:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image3

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image3, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image3,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image3,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image3, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image3,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image3, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)

                        edges3 = canny(image_result)

                        st.subheader("Canny Edge Detection Image")
                        st.image(edges3, caption = "CANNY EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table(edges3)

            # for otsu
            if choice2 == "Otsu Edge Detection":

                #creating of columns
                col1, col2, col3 = st.columns([1,1,1])
                
                with col1:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image1

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image1, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image1,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image1,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image1, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image1,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image1, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)

                        edges1 = otsu(image_result)

                        st.subheader("Otsu Edge Detection Image")
                        st.image(edges1, caption = "OTSU EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table(edges1)
                    
                with col2:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image2

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image2, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image2,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image2,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image2, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image2,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image2, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)
                        
                        edges2 = otsu(image_result)

                        st.subheader("Otsu Edge Detection Image")
                        st.image(edges2, caption = "OTSU EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table(edges2)


                with col3:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image3

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image3, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image3,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image3,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image3, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image3,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image3, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)

                        edges3 = otsu(image_result)

                        st.subheader("Otsu Edge Detection Image")
                        st.image(edges3, caption = "OTSU EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table(edges3)
                
            # for prewitt
            if choice2 == "Prewitt Edge Detection":

                #creating of columns
                col1, col2, col3 = st.columns([1,1,1])
                
                with col1:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image1

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image1, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image1,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image1,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image1, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image1,-1,kernel)

                        edges1 = prewitt(image_result)

                        st.subheader("Prewitt Edge Detection Image")
                        st.image(edges1, caption = "PREWITT EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table(edges1)
                    
                with col2:
                    
                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image2

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image2, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image2,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image2,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image2, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image2,-1,kernel)
                        
                        edges2 = prewitt(image_result)

                        st.subheader("Prewitt Edge Detection Image")
                        st.image(edges2, caption = "PREWITT EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table(edges2)

                with col3:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image3

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image3, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image3,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image3,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image3, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image3,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image3, 'haar')
                        #     cA, image_result = coeffs

                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     image_result = image_result / 255
                        #     st.write(image_result)
                        
                        edges3 = prewitt(image_result)

                        st.subheader("Prewitt Edge Detection Image")
                        st.image(edges3, caption = "PREWITT EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table(edges3)
            
            # for robert
            if choice2 == "Robert Edge Detection":

                #creating of columns
                col1, col2, col3 = st.columns([1,1,1])
                
                with col1:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image1
 
                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image1, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image1,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image1,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image1, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image1,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image1, 'haar')

                        #     cA, image_result = coeffs
                        #     st.write(image_result)

                        #     image_result[image_result < 0] = 0
                        #     image_result[image_result > 0] = 255
                        #     # image_result = np.clip(image_result, 0, 1) 
                        #     st.write(image_result)

                        edges1 = robert(image_result)
                        
                        # edges1[edges1 < 0] = 0
                        # edges1[edges1 > 0] = 255
                        # st.write(edges1)

                        st.subheader("Robert Edge Detection Image")
                        st.image(edges1, caption = "ROBERT EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table(edges1)
                    
                with col2:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image2

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image2, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image2,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image2,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image2, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image2,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image2, 'haar')

                        #     cA, image_result = coeffs
                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     st.write(image_result)
                        
                        edges2 = robert(image_result)

                        st.subheader("Robert Edge Detection Image")
                        st.image(edges2, caption = "ROBERT EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table(edges2)


                with col3:

                    if choice3 != "Select one":

                        if choice3 == "No Filter":
                            image_result = apply_image3

                        if choice3 == "Adaptive":
                            # adaptive
                            image_result = cv2.adaptiveThreshold(apply_image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

                        if choice3 == "Median":
                            # median
                            image_result = cv2.medianBlur(apply_image3, 3)

                        if choice3 == "Gaussian":
                            # gaussian filter
                            image_result = cv2.GaussianBlur(apply_image3,(5,5),0)

                        if choice3 == "Bilateral":
                            # bilateral 
                            image_result = cv2.bilateralFilter(apply_image3,9,75,75)

                        if choice3 == "Morphological":
                            # morphological operation
                            kernel = np.ones((5, 5), np.uint8)
                            image_result = cv2.morphologyEx(apply_image3, cv2.MORPH_OPEN, kernel)

                        if choice3 == "Averaging":
                            # averaging filter
                            kernel = np.ones((5,5),np.float32)/25
                            image_result = cv2.filter2D(apply_image3,-1,kernel)

                        # if choice3 == "DWT":
                        #     # DWT
                        #     coeffs = pywt.dwt(apply_image3, 'haar')

                        #     cA, image_result = coeffs
                        #     st.write(image_result)

                        #     image_result = np.clip(image_result, 0.0, 1.0) 
                        #     st.write(image_result)

                        edges3 = robert(image_result)

                        st.subheader("Robert Edge Detection Image")
                        st.image(edges3, caption = "ROBERT EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            if on4:
                                histogram(image_result)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table(edges3)

# **************************************************************************************************************************************************************
