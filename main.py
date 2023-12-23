# Paper 

# Importing Packages
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import streamlit as st 
import csv
# import pywt

# All feature matrics functions
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
    
    if choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
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

    if choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
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
    
    if choice2 != "Canny Edge Detection" and choice2 != "Otsu Edge Detection":
        # # dictionary of lists 
        # dict = {"Blue": rms_total[0], "Green": rms_total[1], "Red": rms_total} 
        # rms_total_chart = pd.DataFrame(dict)
        rms_total_chart = pd.DataFrame(rms_total,columns=["Blue", "Green", "Red"])
    else:
        rms_total_chart = pd.DataFrame(rms_total)

    st.line_chart(rms_total_chart)
    return rms_total

# def mse(image):
#     mask_size = 2
#     mse_total = []

#     for c in range (0,image.shape[1]-mask_size+1):
#         # mse = []
#         total = 0
#         count = 0
#         for r in range (0,image.shape[0]-mask_size+1):
#             s = np.sum(image[r:r+mask_size,c:c+mask_size])
#             d = mask_size * mask_size
#             avg = s/d
#             temp2 = 0
#             for i in range(0,mask_size):
#                 for j in range(0,mask_size): 
#                     #pixel_b, pixel_g, pixel_r = image[r+i][c+j]
#                     #t = (pixel_b + pixel_g + pixel_r) / 3
#                     t = image[r+i][c+j]
#                     temp2 = temp2 + (t-avg)**2
#             temp = temp2/d
#             total = total + temp
#             count = count + 1
#             # mse.append(temp)
#         total = total / count
#         mse_total.append(total) 
#         # mse.append(total)
#         # with open("mse.csv", 'a', newline='') as file:
#         #     writer = csv.writer(file)
#         #     writer.writerow(mse)

#     st.write("Mean Square Error Line Chart")
#     mse_total_chart = pd.DataFrame(mse_total)

    # if type(mse_total) == list:
    #         mse_total_chart = pd.DataFrame(mse_total,columns=["Blue", "Green", "Red"])
    #     else:
    #         mse_total_chart = pd.DataFrame(mse_total)

    # st.line_chart(mse_total_chart)

#     return mse_total

def table():

    if choice4 == "HUE Coloration":
        # dictionary of lists 
        dict = {"Mean": avg_total, "STD": std_total, "Var": var_total, "RMS": rms_total} # "MSE": mse_total
        df = pd.DataFrame(dict)
        st.subheader("Table")
        st.dataframe(df)
        st.subheader("Combined Graph")
        st.line_chart(df)

# **************************************************************************************************************************************
# All Edge Detection
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

# **************************************************************************************************************************************
# title
st.title("Run Simulation for Bearing Fault Detection")

# sidebar
with st.sidebar:

    choice_img = st.selectbox("Image set", ["Select one", "ImageSet1", "ImageSet2", "ImageSet3", "ImageSet4", "ImageSet5"])

    if choice_img != "Select one":
        choice4 = st.selectbox("Image Coloration", ["Select one", "Original Image", "HUE Coloration", "Pseudo Coloration"])

        if choice4 != "Select one":
            if choice4 == "Pseudo Coloration":
                choice5 = st.selectbox("Types of Pseudo Coloration", ["Select one", "Spring", "Hot", "Cool", "Rainbow", "HSV", "JET"])

                if choice5 != "Select one":
                    choice1 = st.selectbox("Bearing Fault Detection", ["Select one", "Edge Detection", "Edge Detection with filters"])
                    
                    if choice1 == "Edge Detection":
                        if choice4 == "HUE Coloration":
                            choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                        else:
                            choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])

                    if choice1 == "Edge Detection with filters":
                        if choice4 == "HUE Coloration":
                            choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                            choice3 = st.selectbox("Filters", ["Select one", "Adaptive", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])
                        else:
                            choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                            choice3 = st.selectbox("Filters", ["Select one", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])

            if choice4 != "Pseudo Coloration":
                choice1 = st.selectbox("Bearing Fault Detection", ["Select one", "Edge Detection", "Edge Detection with filters"])
                
                if choice1 == "Edge Detection":
                    if choice4 == "HUE Coloration":
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                    else:
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])

                if choice1 == "Edge Detection with filters":
                    if choice4 == "HUE Coloration":
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                        choice3 = st.selectbox("Filters", ["Select one", "Adaptive", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])
                    else:
                        choice2 = st.selectbox("Edge Detection", ["Select one", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                        choice3 = st.selectbox("Filters", ["Select one", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])    

        
#creating of columns
col1, col2, col3 = st.columns([1,1,1])

# ***************************************************************************************************************************************
if choice_img != "Select one":

    st.header('', divider='rainbow')

    # Image sets
    if choice_img == "ImageSet1":
        with col1: 
            img1 = cv2.imread("Experiment12/Noload/004.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("No Load Image1")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)

        with col2:
            img2 = cv2.imread("Experiment12/A30/134.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Image1")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)

        with col3:
            img3 = cv2.imread("Experiment12/A50/257.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Image1")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)

    if choice_img == "ImageSet2":
        with col1:
            img1 = cv2.imread("Experiment12/Noload/005.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("No Load Image2")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)

        with col2:
            img2 = cv2.imread("Experiment12/A30/135.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Image2")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
        
        with col3:
            img3 = cv2.imread("Experiment12/A50/258.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Image2")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)

    if choice_img == "ImageSet3":
        with col1:
            img1 = cv2.imread("Experiment12/Noload/006.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("No Load Image3")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)

        with col2:
            img2 = cv2.imread("Experiment12/A30/136.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Image3")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)

        with col3:
            img3 = cv2.imread("Experiment12/A50/259.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Image3")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)

    if choice_img == "ImageSet4":
        with col1:
            img1 = cv2.imread("Experiment12/Noload/007.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("No Load Image4")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)

        with col2:
            img2 = cv2.imread("Experiment12/A30/137.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Image4")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)

        with col3:
            img3 = cv2.imread("Experiment12/A50/260.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Image4")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)

    if choice_img == "ImageSet5":
        with col1:
            img1 = cv2.imread("Experiment12/Noload/008.bmp")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            st.subheader("No Load Image5")
            st.image(img1, caption = "NO LOAD Image")
            st.write("Image dimensions:", img1.shape)

        with col2:
            img2 = cv2.imread("Experiment12/A30/138.bmp")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            st.subheader("30% Load Image5")
            st.image(img2, caption = "30% LOAD Image")
            st.write("Image dimensions:", img2.shape)

        with col3:
            img3 = cv2.imread("Experiment12/A50/261.bmp")
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            st.subheader("50% Load Image5")
            st.image(img3, caption = "50% LOAD Image")
            st.write("Image dimensions:", img3.shape)

    # ***************************************************************************************************************************************
    if choice4 == "Original Image":

        with col1:

            # Display the hue image
            st.subheader("Original - No Load Image")
            st.image(img1, caption = "ORIGINAL NO LOAD Image")
            st.write("Image dimensions:", img1.shape)
            apply_image1 = img1

        with col2:

            # Display the hue image
            st.subheader("Original - 30% Load Image")
            st.image(img2, caption = "ORIGINAL 30% LOAD Image")
            st.write("Image dimensions:", img2.shape)
            apply_image2 = img2

        with col3:

            # Display the hue image
            st.subheader("Original - 50% Load Image")
            st.image(img3, caption = "ORIGINAL 50% LOAD Image")
            st.write("Image dimensions:", img3.shape)
            apply_image3 = img3

    #for HUE Coloration
    if choice4 == "HUE Coloration":
            
        with col1:

            # Convert BGR image to HSV
            hsv_image1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

            # Extract the hue channel
            hue_image1 = hsv_image1[:, :, 0]  # Hue channel is the first channel in HSV

            # Display the hue image
            st.subheader("HUE Coloration - No Load Image")
            st.image(hue_image1, caption = "HUE NO LOAD Image")
            st.write("Image dimensions:", hue_image1.shape)
            apply_image1 = hue_image1

        with col2:

            # Convert BGR image to HSV
            hsv_image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

            # Extract the hue channel
            hue_image2 = hsv_image2[:, :, 0]  # Hue channel is the first channel in HSV

            # Display the hue image
            st.subheader("HUE Coloration - 30% Load Image")
            st.image(hue_image2, caption = "HUE 30% LOAD Image")
            st.write("Image dimensions:", hue_image2.shape)
            apply_image2 = hue_image2

        with col3:
        
            # Convert BGR image to HSV
            hsv_image3 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV)

            # Extract the hue channel
            hue_image3 = hsv_image3[:, :, 0]  # Hue channel is the first channel in HSV

            # Display the hue image
            st.subheader("HUE Coloration - 50% Load Image")
            st.image(hue_image3, caption = "HUE 50% LOAD Image")
            st.write("Image dimensions:", hue_image3.shape)
            apply_image3 = hue_image3

        #for HUE Coloration
    if choice4 == "Pseudo Coloration":

        if choice5 == "Spring":

            with col1:

                pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_SPRING)

                # Display the hue image
                st.subheader("Pesudo Spring Coloration Image")
                st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                st.write("Image dimensions:", pesudo_image1.shape)
                apply_image1 = pesudo_image1

            with col2:

                pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_SPRING)

                # Display the hue image
                st.subheader("Pesudo Spring Coloration Image")
                st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                st.write("Image dimensions:", pesudo_image2.shape)
                apply_image2 = pesudo_image2

            with col3:
            
                pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_SPRING)

                # Display the hue image
                st.subheader("Pesudo Spring Coloration Image")
                st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                st.write("Image dimensions:", pesudo_image3.shape)
                apply_image3 = pesudo_image3

        if choice5 == "Hot":

            with col1:

                pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_HOT)

                # Display the hue image
                st.subheader("Pesudo Hot Coloration Image")
                st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                st.write("Image dimensions:", pesudo_image1.shape)
                apply_image1 = pesudo_image1

            with col2:

                pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_HOT)

                # Display the hue image
                st.subheader("Pesudo Hot Coloration Image")
                st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                st.write("Image dimensions:", pesudo_image2.shape)
                apply_image2 = pesudo_image2

            with col3:
            
                pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_HOT)

                # Display the hue image
                st.subheader("Pesudo Hot Coloration Image")
                st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                st.write("Image dimensions:", pesudo_image3.shape)
                apply_image3 = pesudo_image3

        if choice5 == "Cool":

            with col1:

                pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_COOL)

                # Display the hue image
                st.subheader("Pesudo Cool Coloration Image")
                st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                st.write("Image dimensions:", pesudo_image1.shape)
                apply_image1 = pesudo_image1

            with col2:

                pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_COOL)

                # Display the hue image
                st.subheader("Pesudo Cool Coloration Image")
                st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                st.write("Image dimensions:", pesudo_image2.shape)
                apply_image2 = pesudo_image2

            with col3:
            
                pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_COOL)

                # Display the hue image
                st.subheader("Pesudo Cool Coloration Image")
                st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                st.write("Image dimensions:", pesudo_image3.shape)
                apply_image3 = pesudo_image3

        if choice5 == "Rainbow":

            with col1:

                pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_RAINBOW)

                # Display the hue image
                st.subheader("Pesudo Rainbow Coloration Image")
                st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                st.write("Image dimensions:", pesudo_image1.shape)
                apply_image1 = pesudo_image1

            with col2:

                pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_RAINBOW)

                # Display the hue image
                st.subheader("Pesudo Rainbow Coloration Image")
                st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                st.write("Image dimensions:", pesudo_image2.shape)
                apply_image2 = pesudo_image2

            with col3:
            
                pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_RAINBOW)

                # Display the hue image
                st.subheader("Pesudo Rainbow Coloration Image")
                st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                st.write("Image dimensions:", pesudo_image3.shape)
                apply_image3 = pesudo_image3

        if choice5 == "HSV":
            
            with col1:

                pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_HSV)

                # Display the hue image
                st.subheader("Pesudo HSV Coloration Image")
                st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                st.write("Image dimensions:", pesudo_image1.shape)
                apply_image1 = pesudo_image1

            with col2:

                pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_HSV)

                # Display the hue image
                st.subheader("Pesudo HSV Coloration Image")
                st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                st.write("Image dimensions:", pesudo_image2.shape)
                apply_image2 = pesudo_image2

            with col3:
            
                pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_HSV)

                # Display the hue image
                st.subheader("Pesudo HSV Coloration Image")
                st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                st.write("Image dimensions:", pesudo_image3.shape)
                apply_image3 = pesudo_image3

        if choice5 == "JET":

            with col1:

                pesudo_image1 = cv2.applyColorMap(img1, cv2.COLORMAP_JET)

                # Display the hue image
                st.subheader("Pesudo JET Coloration Image")
                st.image(pesudo_image1, caption = "PESUDO NO LOAD Image")
                st.write("Image dimensions:", pesudo_image1.shape)
                apply_image1 = pesudo_image1

            with col2:

                pesudo_image2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)

                # Display the hue image
                st.subheader("Pesudo JET Coloration Image")
                st.image(pesudo_image2, caption = "PESUDO 30% LOAD Image")
                st.write("Image dimensions:", pesudo_image2.shape)
                apply_image2 = pesudo_image2

            with col3:
            
                pesudo_image3 = cv2.applyColorMap(img3, cv2.COLORMAP_JET)

                # Display the hue image
                st.subheader("Pesudo JET Coloration Image")
                st.image(pesudo_image3, caption = "PESUDO 50% LOAD Image")
                st.write("Image dimensions:", pesudo_image3.shape)
                apply_image3 = pesudo_image3

    # ****************************************************************************************************************************************
    if choice_img != "Select one" and choice4 != "Select one":

        if choice4 == "Pseudo Coloration" and choice5 == "Select one":
            pass 

        else:

            # Edge Detection 
            if choice1 == "Edge Detection":

                #for Canny
                if choice2 == "Canny Edge Detection":

                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        edges1 = canny(apply_image1)

                        # Display the hue image
                        st.subheader("Canny Edge Detection Image")
                        st.image(edges1, caption = "CANNY EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        edges2 = canny(apply_image2)

                        # Display the hue image
                        st.subheader("Canny Edge Detection Image")
                        st.image(edges2, caption = "CANNY EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table()


                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        edges3 = canny(apply_image3)

                        # Display the hue image
                        st.subheader("Canny Edge Detection Image")
                        st.image(edges3, caption = "CANNY EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table()

                #for otsu
                if choice2 == "Otsu Edge Detection":

                    #creating of columns
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:
                        
                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        edges1 = otsu(apply_image1)

                        # Display the hue image
                        st.subheader("Otsu Edge Detection Image")
                        st.image(edges1, caption = "OTSU EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        edges2 = otsu(apply_image2)

                        # Display the hue image
                        st.subheader("Otsu Edge Detection Image")
                        st.image(edges2, caption = "OTSU EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table()


                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        edges3 = otsu(apply_image3)

                        # Display the hue image
                        st.subheader("Otsu Edge Detection Image")
                        st.image(edges3, caption = "OTSU EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table()
                    
                #for prewitt
                if choice2 == "Prewitt Edge Detection":

                    #creating of columns
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        edges1 = prewitt(apply_image1)
                       
                        # Display the hue image
                        st.subheader("Prewitt Edge Detection Image")
                        st.image(edges1, caption = "PREWITT EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        edges2 = prewitt(apply_image2)

                        # Display the hue image
                        st.subheader("Prewitt Edge Detection Image")
                        st.image(edges2, caption = "PREWITT EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table()

                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        edges3 = prewitt(apply_image3)

                        # Display the hue image
                        st.subheader("Prewitt Edge Detection Image")
                        st.image(edges3, caption = "PREWITT EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table()
                
                #for robert
                if choice2 == "Robert Edge Detection":

                    #creating of columns
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        edges1 = robert(apply_image1)

                        # Display the hue image
                        st.subheader("Robert Edge Detection Image")
                        st.image(edges1, caption = "ROBERT EDGES NO LOAD Image")
                        st.write("Image dimensions:", edges1.shape)

                        avg_total = mean(edges1)
                        std_total = std(edges1)
                        var_total = var(edges1)
                        rms_total = rms(edges1)
                        # mse_total = mse(edges1)
                        table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        edges2 = robert(apply_image2)

                        # Display the hue image
                        st.subheader("Robert Edge Detection Image")
                        st.image(edges2, caption = "ROBERT EDGES 30% LOAD Image")
                        st.write("Image dimensions:", edges2.shape)

                        avg_total = mean(edges2)
                        std_total = std(edges2)
                        var_total = var(edges2)
                        rms_total = rms(edges2)
                        # mse_total = mse(edges2)
                        table()


                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        edges3 = robert(apply_image3)

                        # Display the hue image
                        st.subheader("Robert Edge Detection Image")
                        st.image(edges3, caption = "ROBERT EDGES 50% LOAD Image")
                        st.write("Image dimensions:", edges3.shape)

                        avg_total = mean(edges3)
                        std_total = std(edges3)
                        var_total = var(edges3)
                        rms_total = rms(edges3)
                        # mse_total = mse(edges3)
                        table()

            # ***************************************************************************************************************************************
            # Edge Detection 
            if choice1 == "Edge Detection with filters":

                #for Canny
                if choice2 == "Canny Edge Detection":

                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Canny Edge Detection Image with filter")
                            st.image(edges1, caption = "CANNY EDGES NO LOAD Image")
                            st.write("Image dimensions:", edges1.shape)

                            avg_total = mean(edges1)
                            std_total = std(edges1)
                            var_total = var(edges1)
                            rms_total = rms(edges1)
                            # mse_total = mse(edges1)
                            table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Canny Edge Detection Image with filter")
                            st.image(edges2, caption = "CANNY EDGES 30% LOAD Image")
                            st.write("Image dimensions:", edges2.shape)

                            avg_total = mean(edges2)
                            std_total = std(edges2)
                            var_total = var(edges2)
                            rms_total = rms(edges2)
                            # mse_total = mse(edges2)
                            table()


                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Canny Edge Detection Image with filter")
                            st.image(edges3, caption = "CANNY EDGES 50% LOAD Image")
                            st.write("Image dimensions:", edges3.shape)

                            avg_total = mean(edges3)
                            std_total = std(edges3)
                            var_total = var(edges3)
                            rms_total = rms(edges3)
                            # mse_total = mse(edges3)
                            table()

                #for otsu
                if choice2 == "Otsu Edge Detection":

                    #creating of columns
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Otsu Edge Detection Image with filter")
                            st.image(edges1, caption = "OTSU EDGES NO LOAD Image")
                            st.write("Image dimensions:", edges1.shape)

                            avg_total = mean(edges1)
                            std_total = std(edges1)
                            var_total = var(edges1)
                            rms_total = rms(edges1)
                            # mse_total = mse(edges1)
                            table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Otsu Edge Detection Image with filter")
                            st.image(edges2, caption = "OTSU EDGES 30% LOAD Image")
                            st.write("Image dimensions:", edges2.shape)

                            avg_total = mean(edges2)
                            std_total = std(edges2)
                            var_total = var(edges2)
                            rms_total = rms(edges2)
                            # mse_total = mse(edges2)
                            table()


                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Otsu Edge Detection Image with filter")
                            st.image(edges3, caption = "OTSU EDGES 50% LOAD Image")
                            st.write("Image dimensions:", edges3.shape)

                            avg_total = mean(edges3)
                            std_total = std(edges3)
                            var_total = var(edges3)
                            rms_total = rms(edges3)
                            # mse_total = mse(edges3)
                            table()
                    
                #for prewitt
                if choice2 == "Prewitt Edge Detection":

                    #creating of columns
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Prewitt Edge Detection Image with filter")
                            st.image(edges1, caption = "PREWITT EDGES NO LOAD Image")
                            st.write("Image dimensions:", edges1.shape)

                            avg_total = mean(edges1)
                            std_total = std(edges1)
                            var_total = var(edges1)
                            rms_total = rms(edges1)
                            # mse_total = mse(edges1)
                            table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)
                        
                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Prewitt Edge Detection Image with filter")
                            st.image(edges2, caption = "PREWITT EDGES 30% LOAD Image")
                            st.write("Image dimensions:", edges2.shape)

                            avg_total = mean(edges2)
                            std_total = std(edges2)
                            var_total = var(edges2)
                            rms_total = rms(edges2)
                            # mse_total = mse(edges2)
                            table()

                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Prewitt Edge Detection Image with filter")
                            st.image(edges3, caption = "PREWITT EDGES 50% LOAD Image")
                            st.write("Image dimensions:", edges3.shape)

                            avg_total = mean(edges3)
                            std_total = std(edges3)
                            var_total = var(edges3)
                            rms_total = rms(edges3)
                            # mse_total = mse(edges3)
                            table()
                
                #for robert
                if choice2 == "Robert Edge Detection":

                    #creating of columns
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist1 = cv2.calcHist([apply_image1],[2],None,[256],[0,500])
                            st.line_chart(hist1)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Robert Edge Detection Image with filter")
                            st.image(edges1, caption = "ROBERT EDGES NO LOAD Image")
                            st.write("Image dimensions:", edges1.shape)

                            avg_total = mean(edges1)
                            std_total = std(edges1)
                            var_total = var(edges1)
                            rms_total = rms(edges1)
                            # mse_total = mse(edges1)
                            table()
                        
                    with col2:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist2 = cv2.calcHist([apply_image2],[2],None,[256],[0,500])
                            st.line_chart(hist2)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Robert Edge Detection Image with filter")
                            st.image(edges2, caption = "ROBERT EDGES 30% LOAD Image")
                            st.write("Image dimensions:", edges2.shape)

                            avg_total = mean(edges2)
                            std_total = std(edges2)
                            var_total = var(edges2)
                            rms_total = rms(edges2)
                            # mse_total = mse(edges2)
                            table()


                    with col3:

                        if choice4 != "HUE Coloration":
                            st.subheader("Histogram")
                            hist3 = cv2.calcHist([apply_image3],[2],None,[256],[0,500])
                            st.line_chart(hist3)

                        if choice3 != "Select one":

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

                            # Display the hue image
                            st.subheader("Robert Edge Detection Image with filter")
                            st.image(edges3, caption = "ROBERT EDGES 50% LOAD Image")
                            st.write("Image dimensions:", edges3.shape)

                            avg_total = mean(edges3)
                            std_total = std(edges3)
                            var_total = var(edges3)
                            rms_total = rms(edges3)
                            # mse_total = mse(edges3)
                            table()

    # ****************************************************************************************************************************************
