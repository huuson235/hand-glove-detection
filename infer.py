import onnxruntime as ort
import time
import cv2
import numpy as np

# import torch

class YoloNAS_Onnx:
    def __init__(self, onnx_model=""):
        providers = [ "CUDAExecutionProvider", "CPUExecutionProvider" ]

        # Nếu KO chạy dc CUDAExecutionProvider, thì dùng cuda_stream của pytorch bên dưới
        # providers = [(
        #         "CUDAExecutionProvider",
        #         {
        #             "device_id": torch.cuda.current_device(),
        #             "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
        #         }
        #     )
        # ]

        
        self.session = ort.InferenceSession(onnx_model, providers=providers)
        self.inputs = [o.name for o in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.size = 640

    def infer(self, cv2_bgr_img, log=True, draw_on_image=True, threshold=0.5):
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
        for image_index in range(num_detections.shape[0]):
            for i in range(num_detections[image_index, 0]):
                class_id = pred_classes[image_index, i]
                confidence = pred_scores[image_index, i]
                if confidence < threshold:
                    continue
                x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
                H, W = img.shape[:2]
                x_min = int(x_min / self.size * W)
                x_max = int(x_max / self.size * W)
                y_min = int(y_min / self.size * H)
                y_max = int(y_max / self.size * H)

                objs.append((x_min, y_min, x_max, y_max, class_id, confidence))
                if log:
                    print(
                        f"class_id={class_id}, confidence={confidence:0.2}, coords={(x_min, y_min, x_max, y_max)}, dur={round((t1-t0)*1000, 0)}ms"
                    )

                if draw_on_image:
                    if class_id == 0:
                        color = (0, 255, 0)
                        text = f"glove {confidence:0.2}"
                    if class_id == 1:
                        color = (0, 0, 255)
                        text = f"hand {confidence:0.2}"
                    h = y_max - y_min
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    font_scale = 1
                    font_thickness = 2

                    text_size, _ = cv2.getTextSize(
                        text, font, font_scale, font_thickness
                    )
                    text_w, text_h = text_size
                    # cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.rectangle( img, (x_min, y_min - text_h - font_scale - 1), (x_min + text_w, y_min - font_scale), color, -1,)
                    cv2.putText( img, text, (x_min - font_scale, y_min - font_scale), font, font_scale, (0, 0, 0), font_thickness,)

        return objs, img


if __name__ == "__main__":

    cap = cv2.VideoCapture("/mnt/d/data/Dataset/video-hand-glove/1.mp4")
    yolo_glove_hand = YoloNAS_Onnx("./yolo_nas_s-glove-hand.onnx")
    count = 0
    skip_frame = 5

    while True:
        ret, img = cap.read()
        count += 1
        if count < skip_frame:
            continue

        _, img = yolo_glove_hand.infer(img)

        img = cv2.resize(img, (1280, 720))
        cv2.imshow("hell", img)
        cv2.waitKey(1)
        count = 0
