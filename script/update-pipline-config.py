import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


PIPELINE_CONF_PATH = 'mymodels\pipeline-update.config'

config = config_util.get_configs_from_pipeline_file(PIPELINE_CONF_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(PIPELINE_CONF_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 2                                                                                       #Lenght of labels
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\checkpoint\ckpt-0'           # ssdmodels checkpoint
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= 'data\label_map.pbtxt'                                                       # label_map path
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ['data\\train.record']                                # train.record path
pipeline_config.eval_input_reader[0].label_map_path = 'data\label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ['data\\test.record']                               # test.record path

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(PIPELINE_CONF_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

print('Update pipline config done....')