from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
def sobel_filter(im, k_size):

    im = im.astype(np.float)
    width, height, c = im.shape
    if c > 1:
        img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
    else:
        img = im

    assert(k_size == 3 or k_size == 5);

    if k_size == 3:
        kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
        kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)
    else:
        kh = np.array([[-1, -2, 0, 2, 1],
                   [-4, -8, 0, 8, 4],
                   [-6, -12, 0, 12, 6],
                   [-4, -8, 0, 8, 4],
                   [-1, -2, 0, 2, 1]], dtype = np.float)
        kv = np.array([[1, 4, 6, 4, 1],
                   [2, 8, 12, 8, 2],
                   [0, 0, 0, 0, 0],
                   [-2, -8, -12, -8, -2],
                   [-1, -4, -6, -4, -1]], dtype = np.float)

    '''
    gx = img;
    gy = gx;
    g = gx;
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            gy[i][j] = np.floor_divide(( (kv[0][0] * img[i-1][j-1]) + (kv[0][1] * img[i-1][j]) + (kv[0][2] * img[i-1][j+1]) +
                           (kv[1][0] * img[i][j-1]) + (kv[1][1] * img[i][j]) + (kv[1][2] * img[i][j+1]) +
                           (kv[2][0] * img[i+1][j-1]) + (kv[2][1] * img[i+1][j]) + (kv[2][2] * img[i+1][j+1])),16)

    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            gx[i][j] = np.floor_divide(( (kh[0][0] * img[i-1][j-1]) + (kh[0][1] * img[i-1][j]) + (kh[0][2] * img[i-1][j+1]) +
                           (kh[1][0] * img[i][j-1]) + (kh[1][1] * img[i][j]) + (kh[1][2] * img[i][j+1]) +
                           (kh[2][0] * img[i+1][j-1]) + (kh[2][1] * img[i+1][j]) + (kh[2][2] * img[i+1][j+1])),16)
    '''
    gx = signal.convolve2d(img, kh, mode='same', boundary = 'symm', fillvalue=0)
    gy = signal.convolve2d(img, kv, mode='same', boundary = 'symm', fillvalue=0)

    g = np.sqrt(np.add(np.square(gx), np.square(gy)))
    g *= 255.0 / np.max(g)
    plt.figure(1)
    plt.axis('off')
    plt.imshow(g, cmap=plt.cm.gray)
    plt.show()
    #return g

def main():
    img = Image.open("I23.BMP")
    rgb = np.array(img);
    sobel_filter(rgb,3)


if __name__ == "__main__":
    main()
