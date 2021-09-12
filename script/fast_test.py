import numpy as np
import os
from six.moves import urllib
import tarfile
import tensorflow as tf
import cv2
from distutils.version import StrictVersion

import cv2

cam = cv2.VideoCapture(0)

# module level variables ##############################################################################################
PROTOS_DIR = "protos"
MIN_NUM_PY_FILES_IN_PROTOS_DIR = 5

MODEL_NAME = 'rexona_graph'
FROZEN_INFERENCE_GRAPH_LOC = MODEL_NAME + "/" + "frozen_inference_graph.pb"
LABEL_MAP_LOC = "data/object-detection.pbtxt"

MODEL_NAME1 = 'ssd_mobilenet_v1_coco_2018_01_28'
FROZEN_INFERENCE_GRAPH_LOC1 = MODEL_NAME1 + "/" + "frozen_inference_graph.pb"
LABEL_MAP_LOC1 = "data/mscoco_complete_label_map.pbtxt"

TEST_IMAGES_DIR = "test_images"

NUM_CLASSES = 1

#######################################################################################################################
def main():
    print("starting program . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    # now that we've checked for the protoc compile, import the TensorFlow models repo utils content
    from utils import label_map_util
    from utils import visualization_utils as vis_util

   
    if StrictVersion(tf.__version__) < StrictVersion('1.5.0'):
        print('error: Please upgrade your tensorflow installation to v1.5.* or later!')
        return

    # if the frozen inference graph does not exist after the above, show an error message and bail
    if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC):
        print("unable to get / create the frozen inference graph")
        return
    if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC1):
        print("unable to get / create the frozen inference graph")
        return
    # end if

    # load the frozen model into memory
    print("loading frozen model into memory . . .")
    detection_graph = tf.Graph()
    detection_graph1 = tf.Graph()
    try:
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph1.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC1, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # end with
        # end with
    except Exception as e:
        print("error loading the frozen model into memory: " + str(e))
        return
    # end try

    # load the label map
    print("loading label map . . .")
    label_map = label_map_util.load_labelmap(LABEL_MAP_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    label_map1 = label_map_util.load_labelmap(LABEL_MAP_LOC1)
    categories1 = label_map_util.convert_label_map_to_categories(label_map1, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index1 = label_map_util.create_category_index(categories1)

    print("starting object detection . . .")
    with detection_graph.as_default() and detection_graph1.as_default():
        with tf.Session(graph=detection_graph) as sess:
            with tf.Session(graph=detection_graph1) as sess1:
                while 1:
                    ret, image_np = cam.read()     

                    # Definite input and output Tensors for detection_graph
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    image_tensor1 = detection_graph1.get_tensor_by_name('image_tensor:0')
                    detection_boxes1 = detection_graph1.get_tensor_by_name('detection_boxes:0')
                    detection_scores1 = detection_graph1.get_tensor_by_name('detection_scores:0')
                    detection_classes1 = detection_graph1.get_tensor_by_name('detection_classes:0')
                    num_detections1 = detection_graph1.get_tensor_by_name('num_detections:0')

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    (boxes1, scores1, classes1, num1) = sess1.run(
                        [detection_boxes1, detection_scores1, detection_classes1, num_detections1],
                        feed_dict={image_tensor1: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                       np.squeeze(boxes),
                                                                       np.squeeze(classes).astype(np.int32),
                                                                       np.squeeze(scores),
                                                                       category_index,
                                                                       use_normalized_coordinates=True,
                                                                       line_thickness=8)
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                       np.squeeze(boxes1),
                                                                       np.squeeze(classes1).astype(np.int32),
                                                                       np.squeeze(scores1),
                                                                       category_index1,
                                                                       use_normalized_coordinates=True,
                                                                       line_thickness=8)
                    cv2.imshow("result", cv2.resize(image_np,(800,600)))
                    if cv2.waitKey(1) & 0xff == 27:
                        cv2.destroyAllWindows()
                        break

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(PROTOS_DIR):
        print('ERROR: PROTOS_DIR "' + PROTOS_DIR + '" does not seem to exist')
        print('Did you compile protoc into the TensorFlow models repository?')
        return False
    # end if

    # count the number of .py files in the protos directory, there should be many (20+)
    numPyFilesInProtosDir = 0
    for fileName in os.listdir(PROTOS_DIR):
        if fileName.endswith(".py"):
            numPyFilesInProtosDir += 1
        # end if
    # end for

    # if there are not enough .py files in the protos directory then protoc must not have been compiled,
    # so show an error and return False
    if numPyFilesInProtosDir < MIN_NUM_PY_FILES_IN_PROTOS_DIR:
        print('ERROR: less than ' + str(MIN_NUM_PY_FILES_IN_PROTOS_DIR) + ' .py files were found in PROTOS_DIR' + PROTOS_DIR)
        print('Did you compile protoc into the TensorFlow models repository?')
        return False
    # end if

    if not os.path.exists(LABEL_MAP_LOC):
        print('ERROR: LABEL_MAP_LOC "' + LABEL_MAP_LOC + '" does not seem to exist')
        return False
    # end if

    if not os.path.exists(TEST_IMAGES_DIR):
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
