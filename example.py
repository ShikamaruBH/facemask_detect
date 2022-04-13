from facemaskDetect import FacemaskDetect
import cv2


if __name__ == '__main__':
    # Tạo đối tượng nhận dạng với tham số truyền vào là đường dẫn đến model
    fmd = FacemaskDetect(r'facemask_model2.pb')

    # Đưa vào ảnh đã được cắt khuôn mặt với định dạng RGB
    img = cv2.imread(r'imgs/correct_mask.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Giá trị trả về là kết quả nhận dạng và độ chính xác
    res, conf = fmd.detect(img)
    print(res, conf)

    img = cv2.imread(r'imgs/incorrect_mask.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res, conf = fmd.detect(img)
    print(res, conf)

    img = cv2.imread(r'imgs/no_mask.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res, conf = fmd.detect(img)
    print(res, conf)