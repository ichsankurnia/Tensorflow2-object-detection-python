import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

MY_MODEL = 'mymodels'
PIPELINE_CONF = 'mymodels\pipeline.config'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONF)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MY_MODEL, 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


import cv2 
import numpy as np

LABEL_PATH = 'data\label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH)

cap = cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()

    # # Converting the input frame to grayscale
    # gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

    # # Fliping the image as said in question
    # gray_flip = cv2.flip(gray,1)

    # # Combining the two different image frames in one window
    # combined_window = np.hstack([gray,gray_flip])
    
    flip_img = cv2.flip(frame,1)

    image_np = np.array(flip_img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False
                )

    cv2.imshow('Object Detection', cv2.resize(image_np_with_detections, (800,600)))
    # cv2.imshow('Object Detection', image_np_with_detections)
    
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break