import os
import time
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from distutils.version import StrictVersion

# module level variables ##############################################################################################

MODEL_NAME = 'mymodels\export\saved_model'
FROZEN_INFERENCE_GRAPH_PATH = MODEL_NAME + "/" + "saved_model.pb"
LABEL_MAP_PATH = "data\label_map.pbtxt"

#######################################################################################################################

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def main():
    startTime = time.time()
    print("starting program . . .")
   
    if StrictVersion(tf.__version__) < StrictVersion('2.5.0'):
        print('error: Please upgrade your tensorflow installation to v2.5.* or later!')
        return

    # if the frozen inference graph does not exist after the above, show an error message and bail
    if not os.path.exists(FROZEN_INFERENCE_GRAPH_PATH):
        print("unable to get / create the frozen inference graph")
        return
    # end if

    # load the frozen model into memory
    print("loading frozen model into memory . . .")
    try:
        detect_fn = tf.saved_model.load(MODEL_NAME)
    except Exception as e:
        print("error loading the frozen model into memory: " + str(e))
        return
    # end try

    # load the label map
    print("loading label map . . .")
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)
    print(detect_fn, category_index)

    print('\n\n========== Done! load saved_model n label_map in {} seconds. . . ===========\n\n'.format(time.time() - startTime))
    print("starting object detection . . .")
    
    cam = cv2.VideoCapture(1)

    while cam.isOpened():
        ret, frame = cam.read()     

        # image_np = np.array(frame)
        image_np = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        ## Flip horizontally
        image_np = np.fliplr(image_np).copy()

        # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        ## The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # input_tensor = tf.convert_to_tensor(image_np)
        ## The model expects a batch of images, so add an axis with `tf.newaxis`.
        # input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.80,
            agnostic_mode=False
        )

#             # Definite input and output Tensors for detection_graph
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             # Each box represents a part of the image where a particular object was detected.
#             detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             # Each score represent how level of confidence for each of the objects.
#             # Score is shown on the result image, together with the class label.
#             detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             # Actual detection.
#             (boxes, scores, classes, num) = sess.run(
#                 [detection_boxes, detection_scores, detection_classes, num_detections],
#                 feed_dict={image_tensor: image_np_expanded})
#             # Visualization of the results of a detection.
#             vis_util.visualize_boxes_and_labels_on_image_array(image_np,
#                                                                np.squeeze(boxes),
#                                                                np.squeeze(classes).astype(np.int32),
#                                                                np.squeeze(scores),
#                                                                category_index,
#                                                                use_normalized_coordinates=True,
#                                                                line_thickness=8)
#             # h, w = image_np.shape[:2]

#             # position = boxes[0][0]

#             # (xmin, ymin, xmax, ymax) = (position[1]*w, position[0]*h, position[3]*w, position[2]*h)

#             # kordinat = (int(xmin), int(ymin))

#             # roi = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]

#             # print(kordinat) 

        # cv2.imshow("Realtime Detection", cv2.resize(image_np_with_detections,(800,600)))
        cv2.imshow("Realtime Detection", cv2.resize(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB), (800,600)))
#             # print(scores)
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
            break

#######################################################################################################################
if __name__ == "__main__":
    main()
