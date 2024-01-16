# Importing Packages
# **************************************************************************************************************************************************************

import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import streamlit as st 
import csv

# Input 
# **************************************************************************************************************************************************************

def input(image, message):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.subheader(message)
    st.image(image, caption = message)
    st.write("Image shape:", image.shape)

    return image

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

    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
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

    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
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
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
        rms_total_chart = pd.DataFrame(rms_total,columns=["Blue", "Green", "Red"])
    else:
        rms_total_chart = pd.DataFrame(rms_total)

    st.line_chart(rms_total_chart)
    return rms_total

def histogram(image):
    st.subheader("Histogram of each channel")
    
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Compute histograms for each channel
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    # Create a figure for the histograms
    fig, ax = plt.subplots()

    # Plot all histograms together
    ax.plot(hist_b, color='blue', label='Blue')
    ax.plot(hist_g, color='green', label='Green')
    ax.plot(hist_r, color='red', label='Red')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Color Histograms')
    ax.legend()

    # Display the figure in Streamlit
    st.pyplot(fig)

    st.subheader("Histogram Cumulative")
    
    hist = cv2.calcHist([image],[2],None,[256],[0,500])
    st.line_chart(hist)

# def histogram_mean(image):
#     # Split the image into its color channels
#     b, g, r = cv2.split(image)

#     # Compute histograms for each channel
#     hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
#     hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
#     hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

#     add = sum(hist_b) + sum(hist_g) + sum(hist_r)

#     return add/(256*256*256)

def table(image):
    # dictionary of lists 
    dict = {"Mean": avg_total, "STD": std_total, "Var": var_total, "RMS": rms_total} # "MSE": mse_total
    df = pd.DataFrame(dict)
    st.subheader("Table")
    st.dataframe(df)
    
    d = image.shape[1] - 1
    
    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
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
        
    # dict = {"Mean": sum(avg_total), "STD": sum(std_total), "Var": sum(var_total), "RMS": sum(rms_total)} # "MSE": mse_total
    # df = pd.DataFrame(dict)
    # st.subheader("Table")
    # st.dataframe(df)
    # if choice4 == "HUE Coloration" or (choice6 == "Gamma Transformation" and choice4 == "Gray Coloration"):
    #     st.subheader("Combined Graph")
    #     st.line_chart(df)

def summary_table(message1,avg_total1,std_total1,var_total1,rms_total1, message2,avg_total2,std_total2,var_total2,rms_total2, message3,avg_total3,std_total3,var_total3,rms_total3): # ,hist_avg1 hist_avg2, hist_avg3,
    
    if choice_img != "Select one":
        message1 = "No Load"
        if choice_img == "Set1 - 30% Phase1, 50% Phase1":
            message2 = "30% P1"
            message3 = "50% P1"
        if choice_img == "Set2 - 30% Phase2, 50% Phase1":
            message2 = "30% P2"
            message3 = "50% P1"
        if choice_img == "Set3 - 30% Phase3, 50% Phase1":
            message2 = "30% P3"
            message3 = "50% P1"
        if choice_img == "Set4 - 30% Phase1, 50% Phase2":
            message2 = "30% P1"
            message3 = "50% P2"
        if choice_img == "Set5 - 30% Phase2, 50% Phase2":
            message2 = "30% P2"
            message3 = "50% P2"
        if choice_img == "Set6 - 30% Phase3, 50% Phase2":
            message2 = "30% P3"
            message3 = "50% P2"

        if choice7 != "Select one" and choice6 != "Select one" and choice4 != "Select one":
                if choice4 == "Pseudo Coloration" and choice5 == "Select one":
                    pass
                else:
                    if on6:
                        d = 319
                
                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration" and choice2 != "Canny Edge Detection":
                        avg_total1 = (sum(avg_total1)/d)
                        avg_total2 = (sum(avg_total2)/d)
                        avg_total3 = (sum(avg_total3)/d)

                        std_total1 = (sum(sum(std_total1))/d)
                        std_total2 = (sum(sum(std_total2))/d)
                        std_total3 = (sum(sum(std_total3))/d)

                        var_total1 = (sum(sum(var_total1))/d)
                        var_total2 = (sum(sum(var_total2))/d)
                        var_total3 = (sum(sum(var_total3))/d)

                        rms_total1 = (sum(sum(rms_total1))/d)
                        rms_total2 = (sum(sum(rms_total2))/d)
                        rms_total3 = (sum(sum(rms_total3))/d)

                    else:

                        avg_total1 = (sum(avg_total1)/d)
                        avg_total2 = (sum(avg_total2)/d)
                        avg_total3 = (sum(avg_total3)/d)

                        std_total1 = (sum(std_total1))/d
                        std_total2 = (sum(std_total2))/d
                        std_total3 = (sum(std_total3))/d

                        var_total1 = (sum(var_total1))/d
                        var_total2 = (sum(var_total2))/d
                        var_total3 = (sum(var_total3))/d

                        rms_total1 = (sum(rms_total1))/d
                        rms_total2 = (sum(rms_total2))/d
                        rms_total3 = (sum(rms_total3))/d

                    st.subheader("Summary Table")
                    table_row = [message1,message2,message3]
                    table_mean = [avg_total1,avg_total2,avg_total3]
                    # table_hist_avg = [hist_avg1,hist_avg2,hist_avg3]
                    table_std = [std_total1,std_total2,std_total3]
                    table_var = [var_total1,var_total2,var_total3]
                    table_rms = [rms_total1,rms_total2,rms_total3]
                    # dictionary of lists 
                    dict = {"Category": table_row, "Mean": table_mean, "STD": table_std, "Var": table_var, "RMS": table_rms} # "Hist": table_hist_avg
                    df = pd.DataFrame(dict)
                    st.dataframe(df)

def object(image,x,y,width,height):
    cut = image
    edges = canny(cut)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (x,y,width,height)
    cv2.grabCut(cut,edges,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((edges==2)|(edges==0),0,1).astype('uint8')
    img = cut*mask[:,:,np.newaxis]
    st.subheader("Object Image")
    st.image(img, caption = "OBJECT Image")
    st.write("Image shape:", img.shape)

    cut = image

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    st.image(binary_image, caption = "Binary image")
    st.write("Image shape:", binary_image.shape)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    final = cv2.bitwise_and(cut, cut, mask=closing)
    
    st.image(closing, caption = "Morphological image")
    st.write("Image shape:", closing.shape)
    st.image(final, caption = "Final image")
    st.write("Image shape:", final.shape)
    return final

# Transformation
# *************************************************************************************************************************************************************

def notransformation(image, message):
    # Display the hue image
    st.subheader(message)
    st.image(image, caption = message)
    st.write("Image shape:", image.shape)

    return image

def gamma(image,message):

    # Define the gamma value (adjust as needed)
    gamma = 0.1

    # Perform gamma correction
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    st.subheader(message)
    st.image(gamma_corrected, caption = message)
    st.write("Image shape:", gamma_corrected.shape)

    return gamma_corrected

def log(image, message):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

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

    st.subheader(message)
    st.image(log_transformed_image, caption = message)
    st.write("Image shape:", log_transformed_image.shape)

    return log_transformed_image

def inverselog(image, message):
    # Split the original image into its color channels
    b, g, r = cv2.split(image)

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
    st.subheader(message)
    st.image(inv_log_transformed_image, caption = message)
    st.write("Image shape:", inv_log_transformed_image.shape)

    return inv_log_transformed_image

# Coloration
# *************************************************************************************************************************************************************

def nocoloration(image, message):
    st.subheader(message)
    st.image(image, caption = message)
    st.write("Image shape:", image.shape)
    return image

def gray(image, message):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    st.subheader(message)
    st.image(gray_image, caption = message)
    st.write("Image shape:", gray_image.shape)

    return gray_image

def hue(image, message):
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract the hue channel
    hue_image = hsv_image[:, :, 0]  # Hue channel is the first channel in HSV

    # Display the hue image
    st.subheader(message)
    st.image(hue_image, caption = message, channels='HSV', use_column_width=True)
    st.write("Image shape:", hue_image.shape)

    return hue_image

def pseudo_spring(image,message):
    pseudo_spring_image = cv2.applyColorMap(image, cv2.COLORMAP_SPRING)

    st.subheader(message)
    st.image(pseudo_spring_image, caption = message)
    st.write("Image shape:", pseudo_spring_image.shape)

    return pseudo_spring_image

def pseudo_hot(image,message):
    pseudo_hot_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)

    st.subheader(message)
    st.image(pseudo_hot_image, caption = message)
    st.write("Image shape:", pseudo_hot_image.shape)

    return pseudo_hot_image

def pseudo_cool(image,message):
    pseudo_cool_image = cv2.applyColorMap(image, cv2.COLORMAP_COOL)

    st.subheader(message)
    st.image(pseudo_cool_image, caption = message)
    st.write("Image shape:", pseudo_cool_image.shape)

    return pseudo_cool_image

def pseudo_rainbow(image,message):
    pseudo_rainbow_image = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)

    st.subheader(message)
    st.image(pseudo_rainbow_image, caption = message)
    st.write("Image shape:", pseudo_rainbow_image.shape)

    return pseudo_rainbow_image

def pseudo_hsv(image,message):
    pseudo_hsv_image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)

    st.subheader(message)
    st.image(pseudo_hsv_image, caption = message)
    st.write("Image shape:", pseudo_hsv_image.shape)

    return pseudo_hsv_image

def pseudo_jet(image,message):
    pseudo_jet_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    st.subheader(message)
    st.image(pseudo_jet_image, caption = message)
    st.write("Image shape:", pseudo_jet_image.shape)

    return pseudo_jet_image

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
        on5 = st.toggle('Original Image Histogram')
        choice7 = st.selectbox("Operations on", ["Select one", "Whole image", "Object without background"])
        if choice7 != "Select one":
            if choice7 == "Object without background":
                on1 = st.toggle('Original Object Detected Image Histogram')
            choice6 = st.selectbox("Image Transformation", ["Select one", "No Transformation", "Gamma Transformation", "Log Transformation", "Inverse Log Transformation"])

            if choice6 != "Select one":
                on2 = st.toggle('Transformed Image Histogram')
                choice4 = st.selectbox("Image Coloration", ["Select one", "No Coloration Image", "Gray Coloration", "HUE Coloration", "Pseudo Coloration"])

                if choice4 != "Select one":

                    if choice4 == "Pseudo Coloration":
                        choice5 = st.selectbox("Types of Pseudo Coloration", ["Select one", "Spring", "Hot", "Cool", "Rainbow", "HSV", "JET"])
                        if choice5 != "Select one":
                            on3 = st.toggle('Coloration Image Histogram')
                            choice2 = st.selectbox("Edge Detection", ["Select one", "No Edge Detection", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                            if choice2 != 'Select one':
                                if choice2 == "No Edge Detection":
                                    on6 = st.toggle("Features")
                                else:
                                    choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])
                                    on4 = st.toggle('Edge Detection Histogram')
                                    if choice3 != "Select one":
                                        on6 = st.toggle("Features")

                    else:
                        if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                            on3 = st.toggle('Coloration Image Histogram')

                        if choice4 == "HUE Coloration":
                            choice2 = st.selectbox("Edge Detection", ["Select one", "No Edge Detection", "Canny Edge Detection", "Otsu Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                            if choice2 != 'Select one':
                                if choice2 == "No Edge Detection":
                                    on6 = st.toggle("Features")
                                else:
                                    choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"])
                                    if choice3 != "Select one":
                                        on6 = st.toggle("Features")
                        else:
                            choice2 = st.selectbox("Edge Detection", ["Select one", "No Edge Detection", "Canny Edge Detection", "Prewitt Edge Detection", "Robert Edge Detection"])
                            if choice2 != 'Select one':
                                if choice2 == "No Edge Detection":
                                    on6 = st.toggle("Features")
                                else:
                                    choice3 = st.selectbox("Filters", ["Select one", "No Filter", "Median", "Gaussian", "Bilateral", "Morphological", "Averaging"]) 
                                    if choice4 != "Gray Coloration":
                                        on4 = st.toggle('Edge Detection Histogram')
                                    if choice3 != "Select one":
                                        on6 = st.toggle("Features")
                    
# Creating of Columns
col1, col2, col3 = st.columns([1,1,1])

# All Options 
# **************************************************************************************************************************************************************

if choice_img != "Select one":

    st.header('', divider='rainbow')

    # Image sets
    # No Load - (70,49,200,166)
    # 10 Phase 1 - (0,0,319,240)
    # 10 Phase 2 - (0,0,319,240)
    # 10 Phase 3 - (0,0,259,239)
    # 30% Phase 1 - (70,49,200,166)
    # 30% Phase 2 - (100,40,170,185)
    # 30% Phase 3 - (15,40,257,176) 
    # 50% Phase 1 - (15,40,257,170) 
    # 50% Phase 2 - (20,35,260,183)
    if choice_img == "Set1 - 30% Phase1, 50% Phase1":
        with col1: 
            img1 = cv2.imread("Noload/004.bmp")
            original_image1 = input(img1, "Original - No Load Image")
            img1 = original_image1
            if on5:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A30/137.bmp")
            original_image2 = input(img2, "Original - 30% Load Phase 1")
            img2 = original_image2
            if on5:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A50/256.bmp")
            original_image3 = input(img3,"Original - 50% Load Phase 1")
            img3 = original_image3
            if on5:
                histogram(img3)

    if choice_img == "Set2 - 30% Phase2, 50% Phase1":
        with col1:
            img1 = cv2.imread("Noload/005.bmp")
            original_image1 = input(img1, "Original - No Load Image")
            img1 = original_image1
            if on5:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C30/177.bmp")
            original_image2 = input(img2, "Original - 30% Load Phase 2")
            img2 = original_image2
            if on5:
                histogram(img2)
        
        with col3:
            img3 = cv2.imread("A50/258.bmp")
            original_image3 = input(img3, "Original - 50% Load Phase 1")
            img3 = original_image3
            if on5:
                histogram(img3)

    if choice_img == "Set3 - 30% Phase3, 50% Phase1":
        with col1:
            img1 = cv2.imread("Noload/006.bmp")
            original_image1 = input(img1, "Original - No Load Image")
            img1 = original_image1
            if on5:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C&B30/217.bmp")
            original_image2 = input(img2, "Original - 30% Load Phase 3")
            img2 = original_image2
            if on5:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A50/259.bmp")
            original_image3 = input(img3, "Original - 50% Load Phase 1")
            img3 = original_image3
            if on5:
                histogram(img3)

    if choice_img == "Set4 - 30% Phase1, 50% Phase2":
        with col1:
            img1 = cv2.imread("Noload/007.bmp")
            original_image1 = input(img1, "Original - No Load Image")
            img1 = original_image1
            if on5:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A30/134.bmp")
            original_image2 = input(img2, "Original - 30% Load Phase 1")
            img2 = original_image2
            if on5:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A&B50/292.bmp")
            original_image3 = input(img3, "Original - 50% Load Phase 2")
            img3 = original_image3
            if on5:
                histogram(img3)

    if choice_img == "Set5 - 30% Phase2, 50% Phase2":
        with col1:
            img1 = cv2.imread("Noload/008.bmp")
            original_image1 = input(img1, "Original - No Load Image")
            img1 = original_image1
            if on5:
                histogram(img1)


        with col2:
            img2 = cv2.imread("A&C30/180.bmp")
            original_image2 = input(img2, "Original - 30% Load Phase 2")
            img2 = original_image2
            if on5:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A&B50/293.bmp")
            original_image3 = input(img3, "Original - 50% Load Phase 2")
            img3 = original_image3
            if on5:
                histogram(img3)

    if choice_img == "Set6 - 30% Phase3, 50% Phase2":
        with col1:
            img1 = cv2.imread("Noload/009.bmp")
            original_image1 = input(img1, "Original - No Load Image")
            img1 = original_image1
            if on5:
                histogram(img1)

        with col2:
            img2 = cv2.imread("A&C&B30/215.bmp")
            original_image2 = input(img2, "Original - 30% Load Phase 3")
            img2 = original_image2
            if on5:
                histogram(img2)

        with col3:
            img3 = cv2.imread("A&B50/293.bmp")
            original_image3 = input(img3, "Original - 50% Load Phase 2")
            img3 = original_image3
            if on5:
                histogram(img3)
    
    if choice7 != "Select one":

        if choice7 == "Whole Image":
            pass

        if choice7 == "Object without background":

            if choice_img == "Set1 - 30% Phase1, 50% Phase1":
                with col1: 
                    cut1 = original_image1
                    img1 = object(cut1,70,49,200,166)
                    if on1:
                        histogram(img1)

                with col2:
                    cut2 = original_image2
                    img2 = object(cut2,70,49,200,166)
                    if on1:
                        histogram(img2)

                with col3:
                    cut3 = original_image3
                    img3 = object(cut3,15,40,257,170)
                    if on1:
                        histogram(img3)

            if choice_img == "Set2 - 30% Phase2, 50% Phase1":
                with col1:
                    cut1 = original_image1 
                    img1 = object(cut1,70,49,200,166)
                    if on1:
                        histogram(img1)

                with col2:
                    cut2 = original_image2
                    img2 = object(cut2,100,40,170,185)
                    if on1:
                        histogram(img2)
                
                with col3:
                    cut3 = original_image3
                    img3 = object(cut3,15,40,257,170)
                    if on1:
                        histogram(img3)

            if choice_img == "Set3 - 30% Phase3, 50% Phase1":
                with col1:
                    cut1 = original_image1 
                    img1 = object(cut1,70,49,200,166)
                    if on1:
                        histogram(img1)

                with col2:
                    cut2 = original_image2
                    img2 = object(cut2,15,40,257,176)
                    if on1:
                        histogram(img2)

                with col3:
                    cut3 = original_image3
                    img3 = object(cut3,15,40,257,170)
                    if on1:
                        histogram(img3)

            if choice_img == "Set4 - 30% Phase1, 50% Phase2":
                with col1:
                    cut1 = original_image1
                    img1 = object(cut1,70,49,200,166)
                    if on1:
                        histogram(img1)

                with col2:
                    cut2 = original_image2
                    img2 = object(cut2,70,49,200,166)
                    if on1:
                        histogram(img2)

                with col3:
                    cut3 = original_image3
                    img3 = object(cut3,20,35,260,183)
                    if on1:
                        histogram(img3)

            if choice_img == "Set5 - 30% Phase2, 50% Phase2":
                with col1:
                    cut1 = original_image1 
                    img1 = object(cut1,70,49,200,166)
                    if on1:
                        histogram(img1)

                with col2:
                    cut2 = original_image2
                    img2 = object(cut2,100,40,170,185)
                    if on1:
                        histogram(img2)

                with col3:
                    cut3 = original_image3
                    img3 = object(cut3,20,35,260,183)
                    if on1:
                        histogram(img3)

            if choice_img == "Set6 - 30% Phase3, 50% Phase2":
                with col1:
                    cut1 = original_image1 
                    img1 = object(cut1,70,49,200,166)
                    if on1:
                        histogram(img1)

                with col2:
                    cut2 = original_image2
                    img2 = object(cut2,15,40,257,176)
                    if on1:
                        histogram(img2)

                with col3:
                    cut3 = original_image3
                    img3 = object(cut3,20,35,260,183)
                    if on1:
                        histogram(img3)
                
        if choice6 != "Select one":

            # Image Transformation
            # **************************************************************************************************************************************************************

            if choice6 == "No Transformation":
                with col1:

                    img1 = notransformation(img1, "No Transformation - No Load Image")
                    transformed_image1 = img1
                    if on2:
                        histogram(transformed_image1)

                with col2:

                    img2 = notransformation(img2, "No Transformation - 30% Load Image")
                    transformed_image2 = img2
                    if on2:
                        histogram(transformed_image2)

                with col3:

                    img3 = notransformation(img3, "No Transformation - 50% Load Image")
                    transformed_image3 = img3
                    if on2:
                        histogram(transformed_image3)

            if choice6 == "Gamma Transformation":

                with col1:
                    gamma_corrected1 = gamma(img1, "Gamma Transformation - No Load Image")
                    transformed_image1 = gamma_corrected1
                    if on2:
                        histogram(transformed_image1)

                with col2:

                    # Perform gamma correction
                    gamma_corrected2 = gamma(img2, "Gamma Transformation - 30% LOAD Image")
                    transformed_image2 = gamma_corrected2
                    if on2:
                        histogram(transformed_image2)

                with col3:

                    gamma_corrected3 = gamma(img3,"Gamma Transformation - 50% LOAD Image")
                    transformed_image3 = gamma_corrected3
                    if on2:
                        histogram(transformed_image3)

            if choice6 == "Log Transformation":

                with col1:

                    log_transformed_image1 = log(img1, "Log Transformation --- No Load Image")          
                    transformed_image1 = log_transformed_image1
                    if on2:
                        histogram(transformed_image1)

                with col2:
                    
                    log_transformed_image2 = log(img2, "Log Transformation - 30% LOAD Image")
                    transformed_image2 = log_transformed_image2
                    if on2:
                        histogram(transformed_image2)

                with col3:
                    
                    log_transformed_image3 = log(img3, "Log Transformation - 50% LOAD Image")
                    transformed_image3 = log_transformed_image3
                    if on2:
                        histogram(transformed_image3)

            if choice6 == "Inverse Log Transformation":
                
                with col1:
                    inv_log_transformed_image1 = inverselog(img1, "Inverse Log Transformation - NO LOAD Image")
                    transformed_image1 = inv_log_transformed_image1
                    if on2:
                        histogram(transformed_image1)

                with col2:
                    
                    inv_log_transformed_image2 = inverselog(img2, "Inverse Log Transformation - 30% LOAD Image")
                    transformed_image2 = inv_log_transformed_image2
                    if on2:
                        histogram(transformed_image2)

                with col3:
                    
                    inv_log_transformed_image3 = inverselog(img3,"Inverse Log Transformation - 50% LOAD Image")
                    transformed_image3 = inv_log_transformed_image3
                    if on2:
                        histogram(transformed_image3)

            # Image Coloration
            # **************************************************************************************************************************************************************
                        
            if choice4 != "Select one":

                if choice4 == "No Coloration Image":

                    with col1:

                        transformed_image1 = nocoloration(transformed_image1, "No Coloration - No Load Image")
                        apply_image1 = transformed_image1
                        if on3:
                            histogram(apply_image1)

                    with col2:

                        transformed_image2 = nocoloration(transformed_image2, "No Coloration - 30% Load Image")
                        apply_image2 = transformed_image2
                        if on3:
                            histogram(apply_image2)

                    with col3:

                        transformed_image3 = nocoloration(transformed_image3, "No Coloration - 50% Load Image")
                        apply_image3 = transformed_image3
                        if on3:
                            histogram(apply_image3)

                if choice4 == "Gray Coloration":

                    with col1:

                        gray_image1 = gray(transformed_image1, "Gray Coloration - No Load Image")
                        apply_image1 = gray_image1

                    with col2:

                        gray_image2 = gray(transformed_image2, "Gray Coloration - 30% Load Image")
                        apply_image2 = gray_image2

                    with col3:

                        gray_image3 = gray(transformed_image3, "Gray Coloration - 50% Load Image")
                        apply_image3 = gray_image3

                if choice4 == "HUE Coloration":
                        
                    with col1:

                        hue_image1 = hue(transformed_image1, "HUE Coloration - No Load Image")
                        apply_image1 = hue_image1

                    with col2:

                        hue_image2 = hue(transformed_image2, "HUE Coloration - 30% Load Image")
                        apply_image2 = hue_image2

                    with col3:
                    
                        hue_image3 = hue(transformed_image3, "HUE Coloration - 50% Load Image")
                        apply_image3 = hue_image3

                if choice4 == "Pseudo Coloration":

                    if choice5 == "Select one":
                        pass

                    if choice5 == "Spring":

                        with col1:

                            pseudo_spring_image1 = pseudo_spring(transformed_image1, "Pseudo Spring Coloration - No Load Image")
                            apply_image1 = pseudo_spring_image1
                            if on3:
                                histogram(apply_image1)

                        with col2:

                            pseudo_spring_image2 = pseudo_spring(transformed_image2, "Pseudo Spring Coloration - 30% Load Image")
                            apply_image2 = pseudo_spring_image2
                            if on3:
                                histogram(apply_image2)

                        with col3:
                        
                            pseudo_spring_image3 = pseudo_spring(transformed_image3, "Pseudo Spring Coloration - 50% Load Image")
                            apply_image3 = pseudo_spring_image3
                            if on3:
                                histogram(apply_image3)

                    if choice5 == "Hot":

                        with col1:

                            pseudo_hot_image1 = pseudo_hot(transformed_image1, "Pseudo Hot Coloration - No Load Image")
                            apply_image1 = pseudo_hot_image1
                            if on3:
                                histogram(apply_image1)

                        with col2:

                            pseudo_hot_image2 = pseudo_hot(transformed_image2, "Pseudo Hot Coloration - 30% Load Image")
                            apply_image2 = pseudo_hot_image2
                            if on3:
                                histogram(apply_image2)

                        with col3:
                        
                            pseudo_hot_image3 = pseudo_hot(transformed_image3, "Pseudo Hot Coloration - 50% Load Image")
                            apply_image3 = pseudo_hot_image3
                            if on3:
                                histogram(apply_image3)

                    if choice5 == "Cool":

                        with col1:

                            pseudo_cool_image1 = pseudo_cool(transformed_image1, "Pseudo Cool Coloration - No Load Image")
                            apply_image1 = pseudo_cool_image1
                            if on3:
                                histogram(apply_image1)

                        with col2:

                            pseudo_cool_image2 = pseudo_cool(transformed_image2, "Pseudo Cool Coloration - 30% Load Image")
                            apply_image2 = pseudo_cool_image2
                            if on3:
                                histogram(apply_image2)

                        with col3:
                        
                            pseudo_cool_image3 = pseudo_cool(transformed_image3, "Pseudo Cool Coloration - 50% Load Image")
                            apply_image3 = pseudo_cool_image3
                            if on3:
                                histogram(apply_image3)

                    if choice5 == "Rainbow":

                        with col1:

                            pseudo_rainbow_image1 = pseudo_rainbow(transformed_image1, "Pseudo Rainbow Coloration - No Load Image")
                            apply_image1 = pseudo_rainbow_image1
                            if on3:
                                histogram(apply_image1)

                        with col2:

                            pseudo_rainbow_image2 = pseudo_rainbow(transformed_image2, "Pseudo Rainbow Coloration - 30% Load Image")
                            apply_image2 = pseudo_rainbow_image2
                            if on3:
                                histogram(apply_image2)

                        with col3:
                        
                            pseudo_rainbow_image3 = pseudo_rainbow(transformed_image3, "Pseudo Rainbow Coloration - 50% Load Image")
                            apply_image3 = pseudo_rainbow_image3
                            if on3:
                                histogram(apply_image3)

                    if choice5 == "HSV":
                        
                        with col1:

                            pseudo_hsv_image1 = pseudo_hsv(transformed_image1, "Pseudo HSV Coloration - No Load Image")
                            apply_image1 = pseudo_hsv_image1
                            if on3:
                                histogram(apply_image1)

                        with col2:

                            pseudo_hsv_image2 = pseudo_hsv(transformed_image2, "Pseudo HSV Coloration - 30% Load Image")
                            apply_image2 = pseudo_hsv_image2
                            if on3:
                                histogram(apply_image2)

                        with col3:
                        
                            pseudo_hsv_image3 = pseudo_hsv(transformed_image3, "Pseudo HSV Coloration - 50% Load Image")
                            apply_image3 = pseudo_hsv_image3
                            if on3:
                                histogram(apply_image3)

                    if choice5 == "JET":

                        with col1:

                            pseudo_jet_image1 = pseudo_jet(transformed_image1, "Pseudo Jet Coloration - No Load Image")
                            apply_image1 = pseudo_jet_image1
                            if on3:
                                histogram(apply_image1)

                        with col2:

                            pseudo_jet_image2 = pseudo_jet(transformed_image2, "Pseudo Jet Coloration - 30% Load Image")
                            apply_image2 = pseudo_jet_image2
                            if on3:
                                histogram(apply_image2)

                        with col3:
                        
                            pseudo_jet_image3 = pseudo_jet(transformed_image3, "Pseudo Jet Coloration - 50% Load Image")
                            apply_image3 = pseudo_jet_image3
                            if on3:
                                histogram(apply_image3)


                if choice4 == 'Pseudo Coloration' and choice5 == "Select one":
                    pass
                        
                else:
                    
                    if choice2 != "Select one":

                        if choice2 == "No Edge Detection":
                            
                            if on6:
                                with col1:
                                    avg_total = mean(apply_image1)
                                    std_total= std(apply_image1)
                                    var_total = var(apply_image1)
                                    rms_total = rms(apply_image1)
                                    table(apply_image1)

                                    avg_total1 = avg_total
                                    std_total1 = std_total
                                    var_total1 = var_total
                                    rms_total1 = rms_total

                                    # hist_avg1 = histogram_mean(apply_image1)

                                with col2:
                                    avg_total = mean(apply_image2)
                                    std_total = std(apply_image2)
                                    var_total = var(apply_image2)
                                    rms_total = rms(apply_image2)
                                    table(apply_image2)

                                    avg_total2 = avg_total
                                    std_total2 = std_total
                                    var_total2 = var_total
                                    rms_total2 = rms_total

                                    # hist_avg2 = histogram_mean(apply_image2)

                                with col3:
                                    avg_total = mean(apply_image3)
                                    std_total = std(apply_image3)
                                    var_total = var(apply_image3)
                                    rms_total = rms(apply_image3)
                                    table(apply_image3)

                                    avg_total3 = avg_total
                                    std_total3 = std_total
                                    var_total3 = var_total
                                    rms_total3 = rms_total

                                    # hist_avg3 = histogram_mean(apply_image3)
                            
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

                                    edges1 = canny(image_result)

                                    st.subheader("Canny Edge Detection Image")
                                    st.image(edges1, caption = "CANNY EDGES NO LOAD Image")
                                    st.write("Image dimensions:", edges1.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)
                                            
                                    if on6:
                                        avg_total = mean(edges1)
                                        std_total = std(edges1)
                                        var_total = var(edges1)
                                        rms_total = rms(edges1)
                                        table(edges1)

                                        avg_total1 = avg_total
                                        std_total1 = std_total
                                        var_total1 = var_total
                                        rms_total1 = rms_total
                                
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

                                    edges2 = canny(image_result)

                                    st.subheader("Canny Edge Detection Image")
                                    st.image(edges2, caption = "CANNY EDGES 30% LOAD Image")
                                    st.write("Image dimensions:", edges2.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges2)
                                        std_total = std(edges2)
                                        var_total = var(edges2)
                                        rms_total = rms(edges2)
                                        table(edges2)

                                        avg_total2 = avg_total
                                        std_total2 = std_total
                                        var_total2 = var_total
                                        rms_total2 = rms_total

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

                                    edges3 = canny(image_result)

                                    st.subheader("Canny Edge Detection Image")
                                    st.image(edges3, caption = "CANNY EDGES 50% LOAD Image")
                                    st.write("Image dimensions:", edges3.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges3)
                                        std_total = std(edges3)
                                        var_total = var(edges3)
                                        rms_total = rms(edges3)
                                        table(edges3)

                                        avg_total3 = avg_total
                                        std_total3 = std_total
                                        var_total3 = var_total
                                        rms_total3 = rms_total

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

                                    edges1 = otsu(image_result)

                                    st.subheader("Otsu Edge Detection Image")
                                    st.image(edges1, caption = "OTSU EDGES NO LOAD Image")
                                    st.write("Image dimensions:", edges1.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)
                                    if on6:
                                        avg_total = mean(edges1)
                                        std_total = std(edges1)
                                        var_total = var(edges1)
                                        rms_total = rms(edges1)
                                        table(edges1)

                                        avg_total1 = avg_total
                                        std_total1 = std_total
                                        var_total1 = var_total
                                        rms_total1 = rms_total
                                
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
                                        
                                    edges2 = otsu(image_result)

                                    st.subheader("Otsu Edge Detection Image")
                                    st.image(edges2, caption = "OTSU EDGES 30% LOAD Image")
                                    st.write("Image dimensions:", edges2.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges2)
                                        std_total = std(edges2)
                                        var_total = var(edges2)
                                        rms_total = rms(edges2)
                                        table(edges2)

                                        avg_total2 = avg_total
                                        std_total2 = std_total
                                        var_total2 = var_total
                                        rms_total2 = rms_total


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

                                    edges3 = otsu(image_result)

                                    st.subheader("Otsu Edge Detection Image")
                                    st.image(edges3, caption = "OTSU EDGES 50% LOAD Image")
                                    st.write("Image dimensions:", edges3.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges3)
                                        std_total = std(edges3)
                                        var_total = var(edges3)
                                        rms_total = rms(edges3)
                                        table(edges3)

                                        avg_total3 = avg_total
                                        std_total3 = std_total
                                        var_total3 = var_total
                                        rms_total3 = rms_total

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

                                    if on6:
                                        avg_total = mean(edges1)
                                        std_total = std(edges1)
                                        var_total = var(edges1)
                                        rms_total = rms(edges1)
                                        # mse_total = mse(edges1)
                                        table(edges1)

                                        avg_total1 = avg_total
                                        std_total1 = std_total
                                        var_total1 = var_total
                                        rms_total1 = rms_total
                                
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

                                    if on6:
                                        avg_total = mean(edges2)
                                        std_total = std(edges2)
                                        var_total = var(edges2)
                                        rms_total = rms(edges2)
                                        table(edges2)

                                        avg_total2 = avg_total
                                        std_total2 = std_total
                                        var_total2 = var_total
                                        rms_total2 = rms_total

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

                                    edges3 = prewitt(image_result)

                                    st.subheader("Prewitt Edge Detection Image")
                                    st.image(edges3, caption = "PREWITT EDGES 50% LOAD Image")
                                    st.write("Image dimensions:", edges3.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges3)
                                        std_total = std(edges3)
                                        var_total = var(edges3)
                                        rms_total = rms(edges3)
                                        table(edges3)

                                        avg_total3 = avg_total
                                        std_total3 = std_total
                                        var_total3 = var_total
                                        rms_total3 = rms_total

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
                                        
                                    edges1 = robert(image_result)

                                    st.subheader("Robert Edge Detection Image")
                                    st.image(edges1, caption = "ROBERT EDGES NO LOAD Image")
                                    st.write("Image dimensions:", edges1.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges1)
                                        std_total = std(edges1)
                                        var_total = var(edges1)
                                        rms_total = rms(edges1)
                                        table(edges1)

                                        avg_total1 = avg_total
                                        std_total1 = std_total
                                        var_total1 = var_total
                                        rms_total1 = rms_total
                                
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
                                    
                                    edges2 = robert(image_result)

                                    st.subheader("Robert Edge Detection Image")
                                    st.image(edges2, caption = "ROBERT EDGES 30% LOAD Image")
                                    st.write("Image dimensions:", edges2.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges2)
                                        std_total = std(edges2)
                                        var_total = var(edges2)
                                        rms_total = rms(edges2)
                                        table(edges2)

                                        avg_total2 = avg_total
                                        std_total2 = std_total
                                        var_total2 = var_total
                                        rms_total2 = rms_total

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

                                    edges3 = robert(image_result)

                                    st.subheader("Robert Edge Detection Image")
                                    st.image(edges3, caption = "ROBERT EDGES 50% LOAD Image")
                                    st.write("Image dimensions:", edges3.shape)

                                    if choice4 != "HUE Coloration" and choice4 != "Gray Coloration":
                                        if on4:
                                            histogram(image_result)

                                    if on6:
                                        avg_total = mean(edges3)
                                        std_total = std(edges3)
                                        var_total = var(edges3)
                                        rms_total = rms(edges3)
                                        table(edges3)

                                        avg_total3 = avg_total
                                        std_total3 = std_total
                                        var_total3 = var_total
                                        rms_total3 = rms_total
                        
                        if choice2 != "No Edge Detection":
                            if choice3 != "Select one":
                                if on6:
                                    summary_table("",avg_total1,std_total1,var_total1,rms_total1, "",avg_total2,std_total2,var_total2,rms_total2, "",avg_total3,std_total3,var_total3,rms_total3)

                        if choice2 == "No Edge Detection":
                            if on6:
                                summary_table("",avg_total1,std_total1,var_total1,rms_total1,"",avg_total2,std_total2,var_total2,rms_total2,"",avg_total3,std_total3,var_total3,rms_total3) # ,hist_avg1 hist_avg2 ,hist_avg3
