# reference: https://github.com/bornreddy/smart-thresholds/blob/master/otsu.py
import numpy as np

class Otsu:

    def total_pix(self,image):
        size = image.shape[1] * image.shape[0]
        return size

    def histogramify(self, image):
        grayscale_array = []
        for w in range(0, image.shape[0]):
            for h in range(0, image.shape[1]):
                intensity = image.item((w, h))
                grayscale_array.append(intensity)

        total_pixels = image.shape[1] * image.shape[0]
        bins = range(0, 257)
        img_histogram = np.histogram(grayscale_array, bins)
        return img_histogram

    def otsu(self, image):
        binarizarion = Otsu();
        hist = binarizarion.histogramify(image)
        total = binarizarion.total_pix(image)
        current_max, threshold = 0, 0
        sumT, sumF, sumB = 0, 0, 0
        for i in range(0, 256):
            sumT += i * hist[0][i]
        weightB, weightF = 0, 0
        varBetween, meanB, meanF = 0, 0, 0
        for i in range(0, 256):
            weightB += hist[0][i]
            weightF = total - weightB
            if weightF == 0:
                break
            sumB += i * hist[0][i]
            sumF = sumT - sumB
            meanB = sumB / weightB
            meanF = sumF / weightF
            varBetween = weightB * weightF
            varBetween *= (meanB - meanF) * (meanB - meanF)
            if varBetween > current_max:
                current_max = varBetween
                threshold = i
        binarizarion.threshold(threshold, image)
        return threshold;

    def threshold(self,t, image):
        intensity_array = []
        for w in range(0,image.shape[1]):
            for h in range(0,image.shape[0]):
                intensity = image.item((h,w))
                if (intensity <= t):
                    x = 0
                else:
                    x = 255
        intensity_array.append(x)
        image.copy(intensity_array)