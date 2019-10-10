import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def region_of_interest(img,vertices):
    mask = np.zeros_like(img)   
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def turning_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    return blur_gray


def edged_image(grayscale_image):
    edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)
    return edges

def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
             
    return np.array([x1, y1, x2, y2])

def average_slope(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else :
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


image = mpimg.imread(r'C:\Users\vatsal\Desktop\testtt.jpg')
ysize = image.shape[0]
xsize = image.shape[1]
line_image = np.copy(image)
lane_image = np.copy(image)



left_bottom = [0,ysize]
right_bottom = [xsize,ysize]
apex = [xsize/2,ysize/2]

region_of_interest_vertices = [left_bottom, right_bottom, apex]

cropped_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))

grayscale_image = turning_grayscale(image)

low_threshold =50
high_threshold =250

edges = edged_image(grayscale_image)
mask = np.zeros_like(edges)


ignore_mask_color = 255

imshape = image.shape
vertices = np.array([[(0,imshape[0]), (imshape[1]/2,imshape[0]/2), (imshape[1],imshape[0])]], dtype = np.int32)
print(vertices)
cv2.fillPoly(mask, [vertices], ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
plt.imshow(masked_edges)

rho = 6
theta = np.pi/60
threshold = 15
min_line_length = 60
max_line_gap = 15
line_image = line_image*0


lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
def display_lines(image, lines):
    height = image.shape
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

color_edges = np.dstack((edges, edges, edges))
averaged_line = average_slope(lane_image,lines)
line_image = display_lines(lane_image, averaged_line)

combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)
plt.imshow(combined_image)
