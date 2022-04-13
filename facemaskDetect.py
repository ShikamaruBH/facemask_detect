import os, cv2, time,tensorflow
import numpy as np

if tensorflow.__version__.startswith('1.'):
    import tensorflow.compat.v1 as tf
    from tensorflow.python.platform import gfile
    tf.disable_v2_behavior()
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

def model_restore_from_pb(pb_path, node_dict,GPU_ratio=None):
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,)
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        for key, value in node_dict.items():
            node = sess.graph.get_tensor_by_name(value)
            node_dict[key] = node
        return sess, node_dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

class FacemaskDetect:
    def __init__(self, model_path):
        self.model_path = model_path
        self.label_dict =  {'correct_mask': 0, 'incorrect_mask': 1, 'no_mask': 2}
        self.GPU_ratio = None
        self.model_shape = [None, 64, 64, 3]
        self.config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True)
        self.node_dict = {'input': 'input:0',
                     'keep_prob': 'keep_prob:0',
                     'prediction': 'prediction:0'}

        self.config.gpu_options.allow_growth = True

        d = time.time()

        self.sess, self.tf_dict = model_restore_from_pb(self.model_path, self.node_dict, self.GPU_ratio)
        print(f'Restore facemask model time: {time.time() - d}s')

    def detect(self,img):
        img = cv2.resize(img, (self.model_shape[2],self.model_shape[1]))
        img = np.expand_dims(img,axis=0)

        feed_dict = {self.node_dict['input']: img,
                     self.node_dict['keep_prob']: 1.0}
        predict = self.sess.run(self.node_dict['prediction'], feed_dict)[0]
        index = np.argmax(predict)

        return list(self.label_dict.keys())[index], predict[index]