import supervision as sv
import time
from enum import Enum
from PIL import Image
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
#from rfdetr import RFDETRKeypointPreview
from rfdetr.assets.coco_classes import COCO_CLASSES


class MODEL_SIZE(Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def infer_obj_detection(model_size=MODEL_SIZE.MEDIUM, device="cpu"):
    #
    # Model Loading
    #
    start = time.time()
    match model_size:
        case MODEL_SIZE.NANO:
            model = RFDETRNano(device=device)
        case MODEL_SIZE.SMALL:
            model = RFDETRSmall(device=device)
        case MODEL_SIZE.LARGE:
            model = RFDETRLarge(device=device)
        case _:
            model = RFDETRMedium(device=device)

    print(f"Model Loaded. Elapsed time: {time.time() - start} [sec]")

    #
    # Inference
    #
    start = time.time()
    detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    print(f"Elapsed Time: {time.time() - start} [sec]")
    Image.fromarray(annotated_image).save("result.jpg")


def infer_seg(model_size=MODEL_SIZE.MEDIUM, device="cpu"):
    #
    # Model Loading
    #
    start = time.time()
    match model_size:
        case MODEL_SIZE.NANO:
            model = RFDETRSegNano(device=device)
        case MODEL_SIZE.SMALL:
            model = RFDETRSegSmall(device=device)
        case MODEL_SIZE.LARGE:
            model = RFDETRSegLarge(device=device)
        case _:
            model = RFDETRSegMedium(device=device)

    print(f"Model Loaded. Elapsed time: {time.time() - start} [sec]")

    #
    # Inference
    #
    start = time.time()
    detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    print(f"Elapsed Time: {time.time() - start} [sec]")
    Image.fromarray(annotated_image).save("result.jpg")


if __name__ == '__main__':
    #device = "cuda:0"
    device = "cpu"
    infer_obj_detection(model_size=MODEL_SIZE.NANO, device=device)
    #infer_seg(model_size=MODEL_SIZE.NANO, device=device)
