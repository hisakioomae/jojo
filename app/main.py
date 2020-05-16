import cv2

input_path = "../input/"
output_path = "../output/"


def detect_template(file_name, img):
    # テンプレートの重ちーを入力
    template = cv2.imread('../template/' + file_name)
    _, w, h = template.shape[::-1]
    # 検出するスマホのスクショを入力
    # テンプレートマッチング
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    btm_right = (top_left[0] + w, top_left[1] + h)
    # マッチングしたものに四角を描く
    cv2.rectangle(img, top_left, btm_right, 255, 2)


# 位置
def main():

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("start!")
    main()
    print("end!")
