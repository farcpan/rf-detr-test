import supervision as sv
from enum import Enum
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
#from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
#from rfdetr import RFDETRKeypointPreview
from rfdetr.assets.coco_classes import COCO_CLASSES


class MODEL_SIZE(Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def infer(model_size=MODEL_SIZE.MEDIUM):
    match model_size:
        case MODEL_SIZE.NANO:
            model = RFDETRNano()
        case MODEL_SIZE.SMALL:
            model = RFDETRSmall()
        case MODEL_SIZE.LARGE:
            model = RFDETRLarge()
        case _:
            model = RFDETRMedium()

    detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)


if __name__ == '__main__':
    infer(model_size=MODEL_SIZE.NANO)