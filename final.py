import numpy as np
import cv2
import math
from tqdm.auto import tqdm
import os
from pydub import AudioSegment

a = 25
quantConst = a / 200

luminance_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
]) / 255 * np.sqrt(a)

def dctPoint(imge, u, v, N):
    result = 0
    
    for x in range(N):
        for y in range(N):
            result += imge[x, y] * math.cos(((2*x + 1)*u*math.pi)/(2*N)) * math.cos(((2*y + 1)*v*math.pi)/(2*N))

    if (u==0) and (v==0):
        result = result/N
    elif (u==0) or (v==0):
        result = (math.sqrt(2.0)*result)/N
    else:
        result = (2.0*result)/N

    return result

def idctPoint(dctImge, x, y, N):
    result = 0

    for u in range(N):
        for v in range(N):
            if (u==0) and (v==0):
                tau = 1.0/N
            elif (u==0) or (v==0):
                tau = math.sqrt(2.0)/N
            else:
                tau = 2.0/N            
            result += tau * dctImge[u, v] * math.cos(((2*x + 1)*u*math.pi)/(2*N)) * math.cos(((2*y + 1)*v*math.pi)/(2*N))

    return result

def dctBlocks(blocks):
    res = []

    for i in tqdm(range(len(blocks)), desc="ДКП блоков"):
        imge = blocks[i]
        N = imge.shape[0]
        dctRes = np.zeros([N, N], dtype=float)
        for u in range(N):
            for v in range(N):
                val = dctPoint(imge, u, v, N)
                dctRes[u, v] = round(val / luminance_table[u, v]) * luminance_table[u, v]
        res.append(dctRes)

    return res

def idctBlocks(blocks):
    res = []

    for i in tqdm(range(len(blocks)), desc="Обратное ДКП блоков"):
        imge = blocks[i]
        N = imge.shape[0]
        idctRes = np.zeros([N, N], dtype=float)
        for x in range(N):
            for y in range(N):
                idctRes[x, y] = idctPoint(imge, x, y, N)
        res.append(idctRes)
        
    return res

def toYCbCr(im):
    yCbCr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    yCbCr[:,:,0] = 0.299 * r + 0.587 * g + 0.114 * b
    yCbCr[:,:,1] = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    yCbCr[:,:,2] = 128+0.5 * r- 0.418688 * g - 0.081312 * b
    return yCbCr / 255

def toRGB(im):
    mult = im * 255
    rgb = np.empty_like(im)
    y   = mult[:,:,0]
    cb  = mult[:,:,1] - 128
    cr  = mult[:,:,2] - 128
    rgb[:,:,0] = y + 1.402 * cr
    rgb[:,:,1] = y - .34414 * cb - .71414 * cr
    rgb[:,:,2] = y + 1.772 * cb
    return rgb / 255

def subsample(img):
    subsampled_img = np.zeros((img.shape[0] // 2, img.shape[1] // 2, 3))

    for i in range(0, img.shape[0], 2):
        for j in range(0, img.shape[1], 2):
            subsampled_img[i//2, j//2, 0] = img[i, j, 0]
            subsampled_img[i//2, j//2, 1] = (img[i, j, 1] + img[i, j+1, 1] + img[i+1, j, 1] + img[i+1, j+1, 1])/4
            subsampled_img[i//2, j//2, 2] = (img[i, j, 2] + img[i, j+1, 2] + img[i+1, j, 2] + img[i+1, j+1, 2])/4

    return subsampled_img

def unsubsample(subsampled_img):
    img = np.zeros((subsampled_img.shape[0] * 2, subsampled_img.shape[1] * 2, 3))

    for i in range(subsampled_img.shape[0]):
        for j in range(subsampled_img.shape[1]):
            img[2*i, 2*j] = subsampled_img[i, j]
            img[2*i + 1, 2*j] = subsampled_img[i, j]
            img[2*i, 2*j + 1] = subsampled_img[i, j]
            img[2*i + 1, 2*j + 1] = subsampled_img[i, j]

    return img
    

BLOCK_SIZE = 8

def toBlocks(arr, originalImageShape):
    blocks = []

    for j in range(0, originalImageShape[1], BLOCK_SIZE):
        for i in range(0, originalImageShape[0], BLOCK_SIZE):
            blocks.append(np.array(arr[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]))

    return blocks

def fromBlocks(blocks, originalImageShape):
    res = np.zeros(originalImageShape)
    
    xw = originalImageShape[1] // BLOCK_SIZE

    for i in range(len(blocks)):
        for j in range(BLOCK_SIZE):
            for k in range(BLOCK_SIZE):
                xc = i % xw
                yc = i // xw
                res[xc * BLOCK_SIZE + j, yc * BLOCK_SIZE + k] = blocks[i][j][k]
            
    return res

def RLE(dct_image):
    res = ""

    prev_pixel = None
    pixels = np.array(dct_image).flatten()

    for pixel in pixels:
        if pixel == prev_pixel:
            count += 1
        else:
            if prev_pixel is not None:
                res += f"{count} {prev_pixel} "
            prev_pixel = pixel
            count = 1
            
    res += f"{count} {prev_pixel}"

    return res

def reverseRLE(rle_string, shape):
    splited = rle_string.split(" ")

    counts = np.array([int(i) for i in splited[::2]])
    values = np.array([np.float64(i) for i in splited[1::2]])

    reconstructed_pixels = np.repeat(values, counts)
    reconstructed_img = reconstructed_pixels.reshape(shape)

    return reconstructed_img

def squeeze(inputFilename, outputFilename):
    with tqdm(total = 10, desc="Сжатие изображения") as pbar:
        img = cv2.imread(inputFilename)

        pbar.update(1)

        ycbcr_img = toYCbCr(img)

        pbar.update(1)

        subsampled = subsample(ycbcr_img)

        pbar.update(1)
        
        y = subsampled[:,:,0]
        cb = subsampled[:,:,1]
        cr = subsampled[:,:,2]

        yblocks = toBlocks(y, y.shape)
        cbblocks = toBlocks(cb, cb.shape)
        crblocks = toBlocks(cr, cr.shape)

        pbar.update(1)

        dct_y_blocks = dctBlocks(yblocks)

        pbar.update(1)

        dct_cb_blocks = dctBlocks(cbblocks)

        pbar.update(1)

        dct_cr_blocks = dctBlocks(crblocks)

        pbar.update(1)

        dct_y = fromBlocks(dct_y_blocks, y.shape)
        dct_cb = fromBlocks(dct_cb_blocks, cb.shape)
        dct_cr = fromBlocks(dct_cr_blocks, cr.shape)

        pbar.update(1)

        rle_dct_y = RLE(dct_y)
        rle_dct_cb = RLE(dct_cb)
        rle_dct_cr = RLE(dct_cr)

        pbar.update(1)

        with open("results/original.txt", 'w') as file:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        file.write(f"{str(img[i, j, k])} ")

        with open(outputFilename, 'w') as file:
            file.writelines([
                str(y.shape[0]), ' ', 
                str(y.shape[1]), '\n', 
                rle_dct_y,  '\n', 
                rle_dct_cb, '\n', 
                rle_dct_cr
            ])

        pbar.update(1)

def unsqueeze(inputFilename, outputFilename):
    with tqdm(total = 10, desc="Разжатие изображения") as pbar:
        with open(inputFilename, 'r') as file:
            str_shape = file.readline()

            str_shape = str_shape.split(' ')
            shape = (int(str_shape[0]), int(str_shape[1]))

            rle_dct_y = file.readline().replace(' \n', '')
            rle_dct_cb = file.readline().replace(' \n', '')
            rle_dct_cr = file.readline().replace(' \n', '')
        
        pbar.update(1)

        dct_y = reverseRLE(rle_dct_y, shape)
        dct_cb = reverseRLE(rle_dct_cb, shape)
        dct_cr = reverseRLE(rle_dct_cr, shape)
        
        pbar.update(1)

        dct_y_blocks = toBlocks(dct_y, shape)
        dct_cb_blocks = toBlocks(dct_cb, shape)
        dct_cr_blocks = toBlocks(dct_cr, shape)

        pbar.update(1)

        idct_y_blocks = idctBlocks(dct_y_blocks)

        pbar.update(1)

        idct_cb_blocks = idctBlocks(dct_cb_blocks)

        pbar.update(1)

        idct_cr_blocks = idctBlocks(dct_cr_blocks)

        pbar.update(1)

        reconstructedY = fromBlocks(idct_y_blocks, shape)
        reconstructedCb = fromBlocks(idct_cb_blocks, shape)
        reconstructedCr = fromBlocks(idct_cr_blocks, shape)

        pbar.update(1)

        reconstructed_ycbcr_img = cv2.merge((np.float32(reconstructedY), np.float32(reconstructedCb), np.float32(reconstructedCr)))

        reconstructed_img = toRGB(reconstructed_ycbcr_img)

        pbar.update(1)

        unsubsampled = unsubsample(reconstructed_img)
        
        pbar.update(1) 

        cv2.imshow('Reconstructed', unsubsampled)
        
        cv2.imwrite(outputFilename, unsubsampled * 255)

        pbar.update(1)

# Open the video file
video_path = "inputs/original.mp4"  # Replace with your video file path
video = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not video.isOpened():
    print("Error opening video file")

# Read frames from the video
while True:
    # Read the next frame
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        break

    print(frame.shape)

    frame = cv2.resize(frame, (480, 270))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video object and close windows
video.release()
cv2.destroyAllWindows()

