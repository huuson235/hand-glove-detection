import onnxruntime as ort
import time
import cv2
import numpy as np
import pretty_errors
import cvzone
import torch

POLYGON = np.array([
[0.423, 0.271],
[0.383, 0.492],
[0.388, 0.564],
[0.288, 0.564],
[0.261, 0.55],
[0.234, 0.31],
[0.323, 0.176],
[0.419, 0.168],
], dtype=np.float32)
warning_png = cv2.imread("warning.png",cv2.IMREAD_UNCHANGED)
warning_png = cv2.resize(warning_png, (0, 0), fx=0.3, fy=0.3)
def draw_polygons(frame, color):
    H, W = frame.shape[:2]
    polygon = POLYGON.copy()
    polygon[:, 0] *= W
    polygon[:, 1] *= H
    polygon = np.array(polygon, dtype=np.int32)
    cv2.polylines(frame, [polygon], True, color, 2)
    temp_point = np.array((0.372*W, 0.299*H), dtype=np.int32)
    cv2.line(frame, polygon[1], temp_point, color, 2)
    cv2.line(frame, polygon[5], temp_point, color, 2)
    cv2.line(frame, polygon[-1], temp_point, color, 2)
    return frame

def is_point_inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, point, False)
    return result > 0

class YoloNAS_Onnx:
    def __init__(self, onnx_model=""):
        providers = [ "CUDAExecutionProvider", "CPUExecutionProvider" ]

        # Nếu KO chạy dc CUDAExecutionProvider, thì dùng cuda_stream của pytorch bên dưới
        providers = [(
                "CUDAExecutionProvider",
                {
                    "device_id": torch.cuda.current_device(),
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                }
            )
        ]

        
        self.session = ort.InferenceSession(onnx_model, providers=providers)
        self.inputs = [o.name for o in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.size = 640

    def infer(self, cv2_bgr_img, log=True, draw_on_image=True, threshold=0.5, polygons=None):
        t0 = time.time()

        img = cv2_bgr_img

        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.size, self.size))
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        predictions = self.session.run(self.outputs, {self.inputs[0]: x})
        t1 = time.time()

        num_detections, pred_boxes, pred_scores, pred_classes = predictions
        objs = []
        draw_polygons(img, (0,255,0))
        for image_index in range(num_detections.shape[0]):
            for i in range(num_detections[image_index, 0]):
                class_id = pred_classes[image_index, i]
                confidence = pred_scores[image_index, i]
                if confidence < threshold:
                    continue
                x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
                x_min /= self.size
                y_min /= self.size
                x_max /= self.size
                y_max /= self.size

                is_inside = is_point_inside_polygon(((x_max+x_min)/2, (y_max+y_min)/2), POLYGON)
                
                H, W = img.shape[:2]
                x_min = int(x_min * W)
                x_max = int(x_max * W)
                y_min = int(y_min * H)
                y_max = int(y_max * H)

                objs.append((x_min, y_min, x_max, y_max, class_id, confidence))
                if log:
                    print(
                        f"class_id={class_id}, confidence={confidence:0.2}, coords={(x_min, y_min, x_max, y_max)}, dur={round((t1-t0)*1000, 0)}ms"
                    )

                if draw_on_image:
                    if class_id == 0:
                        color = (0, 127, 255)
                        # text = f"glove {confidence:0.2}"
                        text = f"glove"
                    if class_id == 1:
                        color = (0, 0, 255)
                        # text = f"hand {confidence:0.2}"
                        text = f"hand"
                        if is_inside:
                            draw_polygons(img, (0, 0, 255))
                            img = cvzone.overlayPNG(img, warning_png, pos=[400, 30])


                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    font_scale = 1
                    font_thickness = 2

                    text_size, _ = cv2.getTextSize(
                        text, font, font_scale, font_thickness
                    )
                    text_w, text_h = text_size

                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.rectangle( img, (x_min, y_min - text_h - font_scale - 1), (x_min + text_w, y_min - font_scale), color, -1,)
                    cv2.putText( img, text, (x_min - font_scale, y_min - font_scale), font, font_scale, (0, 0, 0), font_thickness,)

        return objs, img

def on_click(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(round(x/1280, 3), round(y/720, 3))

if __name__ == "__main__":

    cap = cv2.VideoCapture("/mnt/d/data/Dataset/video-hand-glove/4.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)

    yolo_glove_hand = YoloNAS_Onnx("./yolo-nas-s-v2-ave.onnx")
    count = 0
    skip_frame = 1

    writer = cv2.VideoWriter( f"demo-glove-hand-v3.avi", cv2.VideoWriter_fourcc(*"XVID"), int(fps), (1280, 720))

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_click)
    while True:
        ret, img = cap.read()
        if not ret: break
        # count += 1
        # if count < skip_frame:
            # continue

        _, img = yolo_glove_hand.infer(img, threshold=0.4, log=False)

        img = cv2.resize(img, (1280, 720))
        writer.write(img)
        # cv2.imshow("frame", img)
        cv2.waitKey(1)
        count = 0

    # writer.release()
    print("done")