import tensorflow as tf
import os.path
import sys

sys.path.append('/home/robot/Project/CPM')
from CPM import CPM

slim = tf.contrib.slim

"""
model_dir = 'test_pb/'
model_name = 'mymodel.pb'
source_model = model_dir + 'model.ckpt-0'
"""
# model_dir = '/home/lechatelia/Desktop/Codes/TOF_CPM7/logs/20180829_2114'
model_dir = '/home/lechatelia/Desktop/Codes/TOF_CPM7/log1024/20181024_1606'
source_model = os.path.join(model_dir, 'model.ckpt-99')

def freeze_mobilenet(meta_file):

    tf.reset_default_graph()
    
    #inp = tf.placeholder(tf.float32, [None, img_size, img_size, 3], '/Feature_Extractor/conv1_1/')

    model = CPM(pretrained_model=None,
                stage=5,
                num_decoder=2,
                input_color='GRAY',
                in_size_h=88,
                in_size_w=112,
                cpu_only=False,
                training=False)

    model.BuildModel()



    # is_training = False
    # weight_decay = 0.0
    # arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
    #
    # with slim.arg_scope(arg_scope):
    #     logits, _ = mobilenet_v1.mobilenet_v1(inp, num_classes=num_classes, is_training=is_training, depth_multiplier=factor)
    
    #predictions = tf.contrib.layers.softmax(logits)
    """
    output = tf.identity(model.output, name='CPM/final_output')
    
    output_node_names = 'CPM/final_output'
    """
    
    
    # output = tf.identity(model.hm[-1], name='CPM/final_output_stage%d' %(len(model.hm)))
    # output_node_names = 'CPM/final_output_stage%d' %(len(model.hm))
    output = tf.identity(model.decoded_hm[-1], name='CPM/final_decoded_map%d' % (len(model.decoded_hm)))
    output_node_names = 'CPM/final_decoded_map%d' % (len(model.decoded_hm))
    print (output_node_names)
    
    # output =  tf.cast(model.output[-1] * 255, dtype=tf.uint8, name='CPM/final_output')
    #output = tf.identity(model.output[-1], name='CPM/final_output')
    #output_node_names = 'CPM/final_output'
    
    output_txt_name = output_node_names[4:] + '.pbtxt'
    output_pb_name = output_node_names[4:] + '.pb'

    rest_var = slim.get_variables_to_restore()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        saver = tf.train.Saver(rest_var)
        saver.restore(sess, meta_file)

        # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
        # with tf.gfile.GFile(model_dir+model_name, 'wb') as f:
        #     f.write(output_graph_def.SerializeToString())
        """
        tf.train.write_graph(sess.graph_def,"./","final_output.pbtxt",as_text=True)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(","))
        tf.train.write_graph(output_graph_def,"./","final_output.pb",as_text=False)
        """
        tf.train.write_graph(sess.graph_def,"./",output_txt_name,as_text=True)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(","))
        tf.train.write_graph(output_graph_def,"./",output_pb_name,as_text=False)
        
freeze_mobilenet(source_model)


