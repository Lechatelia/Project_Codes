import tensorflow as tf
import numpy as np
from skimage import transform, io, color
import cv2
import os
from sklearn.preprocessing import minmax_scale

# minMax = MinMaxScaler()

# test_pb_path = 'final_output_stage5.pb'
test_pb_path = 'final_decoded_map1.pb'
raw_root = 'test.jpg'
root = './'
def resize_to_imgsz(hm, img):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :return:
    """
#    assert len(hm.shape) == 4 and len(img.shape) == 4
    ret_map = np.zeros((hm.shape[0], img.shape[1], img.shape[2], hm.shape[-1]), np.float32)
    for n in range(hm.shape[0]):
        for c in range(hm.shape[-1]):
            ret_map[n,:,:,c] = transform.resize(hm[n,:,:,c], img.shape[1: 3])
    return ret_map

def predict_raw_root():
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(test_pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input/img_in:0")
            # out_put = sess.graph.get_tensor_by_name("CPM/final_output_stage5:0")
            out_put = sess.graph.get_tensor_by_name("CPM/final_decoded_map1:0")


            t_img = io.imread(raw_root)
            """
            #input_list.append(cv2.resize(t_img, (in_size_w, in_size_h)))
            _input = np.array(t_img)
    
            _input =_input.reshape((1, 560, 2448, 3))
            """
            t_img = transform.resize(t_img, (172, 224))
            # t_img = color.gray2rgb(t_img)
            t_img = color.rgb2gray(t_img)
            _input = np.array(t_img)
            print (_input.shape)
            _input = _input.reshape((1, 172, 224, 1))

            print('input_shape:', _input.shape)
            # pred_map = sess.run(out_put, feed_dict={input_x: _input / 255.0})[:, -1]
            pred_map = sess.run(out_put, feed_dict={input_x: _input})
            print('output_shape:', pred_map.shape)

            r_pred_map = resize_to_imgsz(np.expand_dims(pred_map[0], 0),
                                         np.expand_dims(_input[0], 0))
            print(r_pred_map.shape)

            v_pred_map = np.squeeze(r_pred_map, axis=0)

            w_pred_map=np.sum(v_pred_map[:,:,1:8],axis=2,keepdims=False)
            # w_pred_map=minmax_scale(w_pred_map)
            for c in range(1, 8):
                saveName = os.path.join(root, 'final_pb_ch%d.jpg'%(c))
                io.imsave(saveName,
                          (v_pred_map[:, :, c] * 255.).astype(np.uint8))


            io.imsave(os.path.join(root, '{name}_hm.jpg'.format(name=raw_root)),
                      (w_pred_map * 255.).astype(np.uint8))

            """
            v_pred_map = np.sum(r_pred_map, axis=3)
    
            saveName = os.path.join(root, 'final_pb' + '-'+ '.jpg')
            io.imsave(saveName,
                      (np.sum(v_pred_map, axis=0) * 255.0).astype(np.uint8))
             """


def predict_dir(dir_name,hm_dir):
    if not os.path.exists(hm_dir):
        os.makedirs(hm_dir)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(test_pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input/img_in:0")
            # out_put = sess.graph.get_tensor_by_name("CPM/final_output_stage5:0")
            out_put = sess.graph.get_tensor_by_name("CPM/final_decoded_map1:0")

            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    print(file)
                    t_img = io.imread(os.path.join(dir_name,file))
                    """
                    #input_list.append(cv2.resize(t_img, (in_size_w, in_size_h)))
                    _input = np.array(t_img)
        
                    _input =_input.reshape((1, 560, 2448, 3))
                    """
                    t_img = transform.resize(t_img, (172, 224))
                    # t_img = color.gray2rgb(t_img)
                    t_img = color.rgb2gray(t_img)
                    _input = np.array(t_img)
                    _input = _input.reshape((1, 172, 224, 1))
                    # pred_map = sess.run(out_put, feed_dict={input_x: _input / 255.0})[:, -1]
                    pred_map = sess.run(out_put, feed_dict={input_x: _input})

                    r_pred_map = resize_to_imgsz(np.expand_dims(pred_map[0], 0),
                                                 np.expand_dims(_input[0], 0))
                    v_pred_map = np.squeeze(r_pred_map, axis=0)
                    w_pred_map = np.sum(v_pred_map[:, :, 1:8], axis=2, keepdims=False)
                    io.imsave(os.path.join(hm_dir, '{name}_hm.jpg'.format(name=file)),
                              (w_pred_map * 255.).astype(np.uint8))


if __name__ == '__main__':
    predict_raw_root()
    # predict_dir("/home/lechatelia/Desktop/Codes/TOF/test","/home/lechatelia/Desktop/Codes/TOF/test_hm")

