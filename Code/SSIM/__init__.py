import sys
import numpy as np
import matplotlib.pyplot as plt
from Binarization import Otsu
#from __future__ import division

'''
    the new database contains a larger number (3000) of test images obtained from 25 reference
    images, 24 types of distortions for each reference image, and 5 levels for each type of
    distortion
'''


class ToGreyScale:
    # create a 2 D matrix for grayscale calculations
    def rgb2gray(self, image):
        grey_scale = ToGreyScale()
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = grey_scale.weightedAverage(image[rownum][colnum])
        return grey

    def weightedAverage(self, pixel):
        # return math.pow(math.pow(0.2126 * pixel[0],2.2) + math.pow(0.7152 * pixel[1],2.2) + math.pow(0.0722 * pixel[2],2.2) ,(1/2.2))
        # return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        # return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
        return (np.average(pixel))

def BW_Image(image):
    binar = Otsu();
    threshold = binar.otsu(image)
    bw_image = np.zeros((image.shape[0], image.shape[1]))
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            if image.item((i, j)) < threshold:
                bw_image[i][j] = 255;# connect component
            else:
                bw_image[i][j] = 0; # background
    #print (threshold)
    return bw_image;


def luminance(img):
    return img.mean();


def contrast(img, mean_value):
    return ((((img - mean_value) ** 2).sum()) / (img.size - 1)) ** (0.5)


def structure(img, mean_value, sigma_value):
    return ((img - mean_value) / sigma_value)


def ssim(img1, img2):
    # The system separates the task of similarity measurement into three comparisons: luminance, contrast and structure
    mux = luminance(img1);
    muy = luminance(img2);
    # print mux,'  ', muy;
    sigx = contrast(img1, mux);
    sigy = contrast(img2, muy);
    # print sigx, '  ',sigy
    stux = structure(img1, mux, sigx);
    stuy = structure(img2, muy, sigy);
    # luminance comparision
    L = 255;
    # K1 << 1
    K1 = 0.0001;
    C1 = ((K1 * L) * (K1 * L))
    K2 = 0.0003;
    C2 = ((K2 * L) * (K2 * L))
    sigxy = (((img1 - mux) * (img2 - muy)).sum()) / (img1.size - 1)
    return ((((2 * mux * muy) + C1) * (2 * sigxy + C2)) / (
    ((mux * mux) + (muy * muy) + C1) * ((sigx * sigx) + (sigy * sigy) + C2)))


def readImageSSIM():

    import pylab
    array_24_distortion = [];
    #f = open('ssim-bw-5.txt','wr')
    try:
        from PIL import Image
        for k in range(1,2):
            if k<10:
              img1 = np.asarray(Image.open('../tid2013/reference_images/I0'+str(k)+'.BMP'))
              temp = '../tid2013/distorted_images/i0' + str(k)
            else:
                img1 = np.asarray(Image.open('../tid2013/reference_images/I' + str(k) + '.BMP'))
                temp = '../tid2013/distorted_images/i' + str(k)
            for j in range(1,25):
                ssim_map = 0.0;
                if j <10:
                    img2 = np.asarray(Image.open(temp+'_0'+str(j)+'_'+str(5)+'.bmp'))
                else:
                    img2 = np.asarray(Image.open(temp+'_'+str(j)+'_'+ str(5) + '.bmp'))
                grey_scale = ToGreyScale()
                grey_img1 = grey_scale.rgb2gray(img1)
                grey_img2 = grey_scale.rgb2gray(img2)
                #bw_img1 = BW_Image(grey_img1)
                #bw_img2 = BW_Image(grey_img2)
                ssim_map += ssim(grey_img1, grey_img2)
                avg_5_distortion = ssim_map;
                array_24_distortion.append(avg_5_distortion)
                #f.write(str(avg_5_distortion)+'\n');
    except Exception, e:
        e = 'Cannot load images' + str(e)
        print >> sys.stderr, e
    #f.close();

    '''
    with open("ssim-bw-5.txt", "r") as ins:
        array = []
        for line in ins:
            array.append(float(line.rstrip('\n')))
    i = 0;
    j = 24;
    array_25_image = []
    std_images = []
    while i < 24:
        array_25_image.append(sum(array[i:array.__len__():j]) / 25)
        std_images.append(np.std(array[i:array.__len__():j],axis = 0,ddof=1));
        i +=1;
    print 'ssim'
    print array_25_image.__len__(), array_25_image
    return array_25_image, std_images, array
    '''
    return array_24_distortion

def readImageRMS():

    import pylab
    import math,operator
    array_24_distortion_rms = [];
    #f = open('rms-bw-5.txt','w')
    try:
        from PIL import Image
        for k in range(1,2):
            if k<10:
              img1 = np.asarray(Image.open('../tid2013/reference_images/I0'+str(k)+'.BMP'))
              temp = '../tid2013/distorted_images/i0' + str(k)
            else:
                img1 = np.asarray(Image.open('../tid2013/reference_images/I' + str(k) + '.BMP'))
                temp = '../tid2013/distorted_images/i' + str(k)
            for j in range(1,25):
                rms = 0.0;
                if j <10:
                    img2 = np.asarray(Image.open(temp+'_0'+str(j)+'_'+str(5)+'.bmp'))
                else:
                    img2 = np.asarray(Image.open(temp+'_'+str(j)+'_'+ str(5) + '.bmp'))
                grey_scale = ToGreyScale()
                grey_img1 = grey_scale.rgb2gray(img1)/255
                grey_img2 = grey_scale.rgb2gray(img2)/255
                tmp =  (np.sum(np.square(grey_img1 - grey_img2), keepdims=False) / (np.size(grey_img1)))
                rms+=np.sqrt(tmp)
                avg_5_distortion_rms = 1.0 - (rms);
                array_24_distortion_rms.append(avg_5_distortion_rms)
                #f.write(str(avg_5_distortion_rms)+'\n');
    except Exception, e:
        e = 'Cannot load images' + str(e)
        print >> sys.stderr, e
    #f.close();
    '''
    with open("rms-bw-5.txt", "r") as ins:
        array = []
        for line in ins:
            array.append(float(line.rstrip('\n')))
    # print array
    i = 0;
    j = 24;
    array_25_image_rms = []
    std_images = []
    while i < 24:
        array_25_image_rms.append(sum(array[i:array.__len__():j]) / 25)
        std_images.append(np.std(array[i:array.__len__():j],axis = 0,ddof=1));
        i += 1;
    print 'rms'
    print array_25_image_rms.__len__(), array_25_image_rms
    return array_25_image_rms, std_images
    '''
    return array_24_distortion_rms

def readImageThreadSSIM():

    import pylab
    array_24_distortion = [];
    #f = open('ssim_thread-bw-5.txt', 'w')
    try:
        from PIL import Image
        for k in range(1, 2):
            if k < 10:
                img1 = np.asarray(Image.open('../tid2013/reference_images/I0' + str(k) + '.BMP'))
                temp = '../tid2013/distorted_images/i0' + str(k)
            else:
                img1 = np.asarray(Image.open('../tid2013/reference_images/I' + str(k) + '.BMP'))
                temp = '../tid2013/distorted_images/i' + str(k)
            for j in range(1, 25):
                ssim_1 = 0.0;
                ssim_2 = 0.0;
                ssim_3 = 0.0;
                ssim_4 = 0.0;
                ssim_map = 0;
                if j < 10:
                    img2 = np.asarray(Image.open(temp + '_0' + str(j) + '_' + str(5) + '.bmp'))
                else:
                    img2 = np.asarray(Image.open(temp + '_' + str(j) + '_' + str(5) + '.bmp'))
                grey_scale = ToGreyScale()
                grey_img1 = grey_scale.rgb2gray(img1)
                grey_img2 = grey_scale.rgb2gray(img2)
                #bw_img1 = BW_Image(grey_img1)
                #bw_img2 = BW_Image(grey_img2)
                #split_grey_img1 = np.vsplit(grey_img1, 2)
                #split_grey_img2 = np.vsplit(grey_img2, 2)
                split_grey_img1 = np.vsplit(grey_img1, 2)
                split_grey_img2 = np.vsplit(grey_img2, 2)
                block1_1 = np.hsplit(split_grey_img1[0],2)
                block1_2 = np.hsplit(split_grey_img1[1],2)
                block2_1 = np.hsplit(split_grey_img2[0],2)
                block2_2 = np.hsplit(split_grey_img2[1],2)
                ssim_1 += ssim(block1_1[0], block2_1[0])
                ssim_2 += ssim(block1_1[1], block2_1[1])
                ssim_3 += ssim(block1_2[0], block2_2[0])
                ssim_4 += ssim(block1_2[1], block2_2[1])
                ssim_map = (ssim_1 + ssim_2 + ssim_3 + ssim_4) / 4
                avg_5_distortion = ssim_map;
                array_24_distortion.append(avg_5_distortion)
                #f.write(str(avg_5_distortion) + '\n');
    except Exception, e:
        e = 'Cannot load images' + str(e)
        print >> sys.stderr, e
    #f.close();
    '''
    with open("ssim_thread-bw-5.txt", "r") as ins:
        array_thread = []
        for line in ins:
            array_thread.append(float(line.rstrip('\n')))
    i = 0;
    j = 24;
    array_25_image_thread = []
    std_images = []
    while i < 24:
        array_25_image_thread.append(sum(array_thread[i:array_thread.__len__():j]) / 25)
        std_images.append(np.std(array_thread[i:array_thread.__len__():j],axis = 0,ddof=1));
        i +=1;
    print 'Window SSIM'
    print array_25_image_thread.__len__(), array_25_image_thread
    return array_25_image_thread, std_images
    '''
    return array_24_distortion


'''
def main():

    ssim , ssim_std, arr = readImageSSIM()
    rms, rms_std = readImageRMS()
    thread,thread_std = readImageThreadSSIM()

    n_groups = 24;
    fig = plt.figure(figsize=(6*3.13,4*3.13))
    ax = fig.add_subplot(111)
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, ssim, bar_width,
                     alpha=opacity,
                     color='b',
                     label='SSIM',
                     yerr=ssim_std,
                     error_kw=dict(elinewidth=2,ecolor='black'))

    rects2 = plt.bar(index + bar_width, thread, bar_width,
                     alpha=opacity,
                     color='r',
                     label='SSIM - Window Split',
                     yerr=thread_std,
                     error_kw=dict(elinewidth=2,ecolor='black'))

    rects3 = plt.bar(index + bar_width+bar_width, rms, bar_width,
                     alpha=opacity,
                     color='g',
                     label='RMS',
                     yerr=rms_std,
                     error_kw=dict(elinewidth=2,ecolor='black'))

    #plt.ylim(0.60,1.00)
    plt.xticks(index + bar_width ,('1', '2', '3', '4','5', '6', '7', '8','9', '10','11', '12', '13', '14','15', '16',
     '17', '18','19', '20','21', '22', '23', '24'))
    plt.xlabel('Avg of 25 Images for 24 Types of Errors')
    plt.ylabel('Probability')
    plt.title('Probability by Matrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_error_5_with_std_otsu");
    #plt.show()
'''

def main():
    arr = readImageThreadSSIM()
    #arr = [0.94249542054777746, 0.95915531201121784, 0.94205744016614501, 0.94131018150300449, 0.79635825643071401, 0.93731837463296597, 0.94437063120899778, 0.7794505640026419, 0.78379092068377076, 0.86033351187771268, 0.79999696789132502, 0.82966766542449566, 0.831940931300584, 0.90065838310672519, 0.94398606087527748, 0.89896089589203798, 0.79994746166530495, 0.99647664631464183, 0.94223829835612183, 0.82307775821294993, 0.86549960271255766, 0.80473813759531354, 0.79953274637617899, 0.78919987533816671];
    print arr
    from PIL import Image
    fig = plt.figure(figsize=(6*3.13,4*3.13))
    for i in range(1,25):
        plt.subplot(4,6,i)
        plt.axis('off')
        if i < 10:
            temp = '../tid2013/distorted_images/i0' + str(1)+ '_0' + str(i) + '_' + str(5) + '.bmp'
        else:
            temp = '../tid2013/distorted_images/i0' + str(1) + '_' + str(i) + '_' + str(5) + '.bmp'
        img = np.asarray(Image.open(temp))
        plt.imshow(img,cmap = 'Greys')
        plt.title('SSIM Thread:' + str(round(arr[i-1], 4)))

    plt.show()



if __name__ == '__main__':
    sys.exit(main())
