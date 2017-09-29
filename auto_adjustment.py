import cv2
import numpy as np
from scipy.signal import find_peaks_cwt


class AutoAdjustment():
    def run(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        peaks = find_peaks_cwt(hist.flatten(), np.arange(1, 50))
        indent_right = peaks[-1]
        indent_left = peaks[0]
        dist = indent_right - indent_left
        alpha = 256.0 / dist
        beta = -(indent_left * alpha)

        result = None
        result = cv2.convertScaleAbs(gray, result, alpha, beta)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    auto_adjustment = AutoAdjustment()
    result = auto_adjustment.run('test.jpg')
    cv2.imshow('image', result)
    cv2.waitKey(0)
