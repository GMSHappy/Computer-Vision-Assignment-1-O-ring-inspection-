import cv2 as cv
import numpy as np
import time
import os

def compute_histogram(grey_img):

    # imagine 256 empty buckets, one for each shade (0 = black, 255 = white)
    hist = np.zeros(256, dtype=np.int64)
    
    # find out how tall and wide the image is
    rows = grey_img.shape[0]
    cols = grey_img.shape[1]
    
    # visit every pixel in the image, one by one 
    for r in range(rows):
        for c in range(cols):

            # check what shade of grey this pixel is (0-255)
            gl = grey_img[r, c]
        
            hist[gl] += 1
    
    return hist

def find_threshold(grey_img):

 # add up all pixel shades and divide by the number of pixels = average shade

 # start with the average shade as our first guess for the dividing line
    T = int(np.mean(grey_img))

    while True:

        above = grey_img[grey_img > T]   # pixels brighter than T (the "light" group)
        below = grey_img[grey_img <= T]  # pixels darker than T  (the "dark" group)

        # safety check - if one group is empty, we cant improve T anymore so stop
        if len(above) == 0 or len(below) == 0:
            break

        # find the average shade of each group
        mu1 = np.mean(above)  # average of the light pixels
        mu2 = np.mean(below)  # average of the dark pixels

        # our new dividing line = halfway between the two averages
        new_T = int((mu1 + mu2) / 2)

        # if T didnt change, we've found the best dividing line so stop
        if new_T == T:
            break
        T = new_T

    return T

def threshold_image(grey_img, T):
    rows = grey_img.shape[0]
    cols = grey_img.shape[1]
    
    # create a blank black image the same size as the original
    binary = np.zeros((rows, cols), dtype=np.uint8)
    
    # visit every pixel
    for r in range(rows):
        for c in range(cols):
            # if this pixel is brighter than our dividing line, make it white
            # otherwise it stays black (already set to 0 above)
            if grey_img[r, c] > T:
                binary[r, c] = 255
    
    # return the new black and white image
    return binary


def dilation(binary, kernel):
    ksize = kernel.shape[0]  # size of the kernel
    pad   = ksize // 2       # how many pixels to stay away from the edge
    rows  = binary.shape[0]
    cols  = binary.shape[1]
    out   = np.zeros((rows, cols), dtype=np.uint8)  

    # visit every pixel that isnt on the border
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            hit = False
            
        
            for i in range(ksize):
                for j in range(ksize):

                    # only check spots where the kernel has a 1

                    if kernel[i, j] == 1:

                        # if ANY neighbour is white, this pixel becomes white

                        if binary[r - pad + i, c - pad + j] == 255:
                            hit = True
            if hit:
                out[r, c] = 255
    return out


def erosion(binary, kernel):
    ksize = kernel.shape[0]  # size of the kernel
    pad   = ksize // 2       # border buffer
    rows  = binary.shape[0]
    cols  = binary.shape[1]
    out   = np.zeros((rows, cols), dtype=np.uint8)  # blank output image

    # visit every pixel that isnt on the border
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            all_fg = True
            
            # slide the kernel over this pixel and check its neighbours
            for i in range(ksize):
                for j in range(ksize):
                    # only check spots where the kernel has a 1
                    if kernel[i, j] == 1:
                        # if ANY neighbour is black, this pixel becomes black
                        if binary[r - pad + i, c - pad + j] == 0:
                            all_fg = False
            if all_fg:
                out[r, c] = 255

    return out


def closing(binary, kernel):

    return erosion(dilation(binary, kernel), kernel)

def connected_components(binary):
    rows   = binary.shape[0]
    cols   = binary.shape[1]
    
    # create a blank label map the same size as the image, to store which ring each pixel belongs to
    labels = np.zeros((rows, cols), dtype=np.int32)
    curlab = 1  # start labelling rings from 1

    for r in range(rows):
        for c in range(cols):
            # if this pixel is white and hasnt been labelled yet, its a new ring
            if binary[r, c] == 255 and labels[r, c] == 0:
                labels[r, c] = curlab
                queue = [(r, c)]  # add it to the queue to explore its neighbours

                # keep going until weve explored the whole ring
                while len(queue) > 0:
                    pr, pc = queue.pop(0)  # grab the next pixel to explore

                    # check all 8 surrounding neighbours of this pixel
                    neighbours = [
                        (pr - 1, pc - 1), (pr - 1, pc), (pr - 1, pc + 1),
                        (pr,     pc - 1),                (pr,     pc + 1),
                        (pr + 1, pc - 1), (pr + 1, pc), (pr + 1, pc + 1)
                    ]

                    for nr, nc in neighbours:
                        # make sure the neighbour is inside the image

                        if 0 <= nr < rows and 0 <= nc < cols:

                            # if its white and unlabelled, its part of this ring 

                            if binary[nr, nc] == 255 and labels[nr, nc] == 0:
                                labels[nr, nc] = curlab  # give it the same label
                                queue.append((nr, nc))   # add it to explore later

                curlab += 1  # finished this ring, next ring gets the next number

    # return the label map where each pixel has a number representing which ring it belongs to (0 = background)
    return labels

def find_largest_region(labels):
    best_label = -1   # will hold the label number of the biggest ring
    best_count = 0    # will hold the size of the biggest ring so far

    num_labels = labels.max()  # how many rings were found in total
    
    # go through every ring one by one
    for lbl in range(1, num_labels + 1):
        count = np.sum(labels == lbl)  # count how many pixels belong to this ring
        
        # if this ring is bigger than the current best, update our best
        if count > best_count:
            best_count = count
            best_label = lbl

    # return the label number of the biggest ring
    return best_label


def region_properties(labels, target):
    area  = 0    # how many pixels are in this ring
    sum_r = 0    # running total of row positions to find the centroid
    sum_c = 0    # running total of col positions 

    # set bounding box edges to worst case values so any real pixel beats them
    min_r = labels.shape[0]   # top edge    
    max_r = 0                 # bottom edge 
    min_c = labels.shape[1]   # left edge   
    max_c = 0                 # right edge  

    rows = labels.shape[0]
    cols = labels.shape[1]

    # visit every pixel in the image
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] == target:   # if this pixel belongs to our ring
                area  += 1               # count it
                sum_r += r               # add its row to the running total
                sum_c += c               # add its col to the running total
                
                # expand the bounding box if this pixel is outside the current box
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c

    # centre of the ring = average row and average col of all its pixels
    centroid_r = sum_r / area
    centroid_c = sum_c / area

    distances = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if labels[r, c] == target:
                # check the 4 direct neighbours
                up    = labels[r - 1, c]
                down  = labels[r + 1, c]
                left  = labels[r, c - 1]
                right = labels[r, c + 1]
                
                # if any neighbour is NOT part of our ring, this is an edge pixel
                if up != target or down != target or left != target or right != target:
                    d = np.sqrt((r - centroid_r) ** 2 + (c - centroid_c) ** 2)
                    distances.append(d)  # save the distance from centre to this edge pixel

    if len(distances) > 0:
        mean_d = np.mean(distances)   # average distance from centre to edge
        std_d  = np.std(distances)    # how much that distance varies
        
        # high circularity = consistent distances = round shape
        # +1 stops division by zero if std is 0
        circ = mean_d / (std_d + 1)
    else:
        circ = 0.0

    # fill ratio = how much of the bounding box is actually filled by the ring
    # a solid square would be 1.0, a ring would be lower
    bbox_area = (max_r - min_r) * (max_c - min_c)
    if bbox_area > 0:
        fill = area / bbox_area
    else:
        fill = 0.0

    return area, centroid_r, centroid_c, min_r, max_r, min_c, max_c, circ, fill


def classify_oring(area, circularity, fill_ratio):

    result = "PASS"  # innocent until proven guilty

    # fail if the o-ring doesn't fill enough of its bounding box (probably broken/incomplete)
    if fill_ratio < 0.38:
        result = "FAIL"

    # fail if the ring too small (probably a speck of dust or noise, or maybe the o-ring is really broken)
    if area < 4000:
        result = "FAIL"

    # fail if the ring too big (something wrong with the image/thresholding)
    if area > 13000:
        result = "FAIL"

    return result


# MAIN LOOP

image_folder = "Orings"
image_files  = sorted(os.listdir(image_folder))

for filename in image_files:

    # skip any file that isnt an image
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filepath   = os.path.join(image_folder, filename)
    img_colour = cv.imread(filepath)        # colour version
    img_grey   = cv.imread(filepath, 0)     # greyscale version for processing

    print("Processing: " + filename)
    t_start = time.time()  # start the stopwatch

    # find the best threshold and convert to black & white
    hist   = compute_histogram(img_grey)
    T      = find_threshold(img_grey)
    print("  Threshold = " + str(T))
    binary = threshold_image(img_grey, T)

    # if more than half the image is white, we probably thresholded the wrong way
    # flip it so the o-ring is white and the background is black
    if np.sum(binary == 255) > (binary.size * 0.5):
        binary = 255 - binary

    # 2 cleanup the image (fill small holes in the o-ring)
    kernel       = np.ones((5, 5), dtype=np.uint8)
    binary_clean = closing(binary, kernel)

    # 3 find all separate white rings and label them with different numbers (1, 2, 3 etc)
    labels = connected_components(binary_clean)

    # 4: pick the biggest ring (should be the o-ring) 
    ring_label = find_largest_region(labels)

    if ring_label == -1:
        print("  No oring found in this image")
        print("")
        continue  # skip to the next image

    #  5 measure the ring roperties 
    area, c_r, c_c, mn_r, mx_r, mn_c, mx_c, circ, fill = region_properties(labels, ring_label)

    #6: decide PASS or FAIL
    result  = classify_oring(area, circ, fill)
    t_end   = time.time()
    elapsed = t_end - t_start  # stop the stopwatch

    print("  Area        = " + str(area))
    print("  Circularity = " + str(round(circ, 3)))
    print("  Fill Ratio  = " + str(round(fill, 3)))
    print("  Result      = " + result)
    print("  Time        = " + str(round(elapsed, 3)) + "s")
    print("")

    #draw the result on the image and display i
    output = img_colour.copy()

    # green box = PASS, red box = FAIL
    if result == "PASS":
        rect_col = (0, 220, 0)
        text_col = (0, 200, 0)
    else:
        rect_col = (0, 0, 220)
        text_col = (0, 0, 200)

    # draw bounding box around the o-ring
    cv.rectangle(output, (mn_c, mn_r), (mx_c, mx_r), rect_col, 2)

    # write PASS or FAIL in big text top-left
    cv.putText(output, result, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, text_col, 3)

    # write the stats underneath in smaller white text
    cv.putText(output, "Time: " + str(round(elapsed, 3)) + "s",
               (10, 72),  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(output, "Fill: " + str(round(fill, 3)),
               (10, 96),  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(output, "Circ: " + str(round(circ, 3)),
               (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # write the filename in small text at the bottom
    cv.putText(output, filename, (10, output.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv.imshow("Oring: " + filename, output)
    cv.waitKey(0)       # wait for a key press before moving to the next image
    cv.destroyAllWindows()