from infer import YoloNAS_Onnx
import cv2
import glob
import os

yolo = YoloNAS_Onnx("./yolo_nas_s-glove-hand.onnx")
paths = glob.glob("/mnt/d/data/Dataset/frames/4/*")
SIZE = 640
for path in paths:
    basename = os.path.basename(path)
    label_name = basename[:-3] + "txt"

    img = cv2.imread(path)
    objs, _ = yolo.infer(img, draw_on_image=False, threshold=0.3)

    with open(f"labels_4/{label_name}", "w+") as f:
        lines = []
        for x1, y1, x2, y2, class_id, _ in objs:
            w = (x2-x1) / SIZE
            h = (y2-y1) / SIZE
            cx = (x2 + x1) / 2 / SIZE
            cy = (y2 + y1) / 2 / SIZE
            line = f"{class_id} {cx} {cy} {w} {h}\n"
            lines.append(line)

        f.writelines(lines)