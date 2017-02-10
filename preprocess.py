import cv2


def normalize(X):
    a = -0.1
    b = 0.1
    x_min = 0
    x_max = 255
    return a + (X - x_min) * (b - a) / (x_max - x_min)


def crop_img(img, offset=0):
    """Crop in image data in a consistant way
    offset - shift the cropped image left/right by offset
    """
    offset = int(offset)
    assert -10 <= offset <= 10
    img = normalize(img)
    # corp image
    img = img[50:130, 10+offset:310+offset]
    # resize image (0.5x)
    img = cv2.resize(img, (150, 40))
    return img
