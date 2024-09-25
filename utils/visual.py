import os
import cv2
from myplots import plot_one_box
import argparse


def match_ext(txt, img_dir):
    file_format = ['.jpg', '.bmp', '.png']
    for i in file_format:
        tmp = os.path.splitext(txt)
        tmp_path = os.path.join(img_dir, os.path.splitext(txt)[0] + i)
        if os.path.exists(tmp_path):
            return tmp_path
    print("🔺 the txt {}'s match file not exist 🔺".format(txt))

    return False


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/sdb1/rjzhang/yolov5/datasetHR/images/train')
    parser.add_argument('--gt_labels_path', type=str, default='/mnt/sdb1/rjzhang/yolov5/datasetHR/labels/train')
    parser.add_argument('--save_dir', type=str, default='/mnt/sdb1/rjzhang/yolov5/datasetHR/visual')
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    data_path = opt.data_path
    gt_labels_path = opt.gt_labels_path
    gt_label_files = os.listdir(gt_labels_path)

    save_dir = opt.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for gt_label_file in gt_label_files:
        img_path = match_ext(gt_label_file, data_path)
        if not img_path:
            break

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        gt_label_path = os.path.join(gt_labels_path, gt_label_file)
        # 对于每一个标注框
        print(gt_label_file)
        with open(gt_label_path, "r") as f_gt:

            lines_gt = f_gt.readlines()

            for line in lines_gt:
                line_split = line.replace("\n", "").split(" ")
                c = line_split[0]

                x_center, y_center, w, h = line_split[1:]
                x_center, y_center, w, h = float(x_center) * width, float(y_center) * height, float(w) * width, float(
                    h) * height
                # cv2.circle(img,(int(x_center),int(y_center)),5,(255,0,0),-1)
                # cv2.imshow('img',img)
                # cv2.waitKey(0)
                w_half, h_half = w / 2, h / 2
                x1, y1, x2, y2 = x_center - w_half, y_center - h_half, x_center + w_half, y_center + h_half
                xyxy = [x1, y1, x2, y2]

                colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]]
                plot_one_box(xyxy, img, colors, c)

        save_path = os.path.join(save_dir, os.path.splitext(gt_label_file)[0] + '.jpg')
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
