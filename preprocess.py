import cv2


def normalize(X):
    a = -0.1
    b = 0.1
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


def preprocess_img(img):
    img = normalize(img)
    # corp image
    img = img[50:130, 0:320]
    # resize image (0.5x)
    img = cv2.resize(img, (160, 40))
    return img
