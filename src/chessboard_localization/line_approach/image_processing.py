import cv2

def resize_with_letterbox(image, target_h, target_w):
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h,
        target_h - new_h - pad_h,
        pad_w,
        target_w - new_w - pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return padded, scale, pad_h, pad_w

def apply_image_processing(image):

    scaled_image, scale, pad_h, pad_w = resize_with_letterbox(
        image, 1024, 1024
    )

    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(
        gray,
        ksize=(3, 3),  # kernel size -> since the image is big this must be big
        sigmaX=1,  # horizontal blur intensity
        sigmaY=1,  # vertical blur intensity
    )

    # bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    processed_image = gaussian

    return scaled_image, processed_image