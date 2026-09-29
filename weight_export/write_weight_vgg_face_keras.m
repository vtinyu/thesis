function a = write_weight_vgg_face_keras( in ,fc)
if(fc == 0)
    % Cấu trúc chuẩn Keras v3: /layers/<tên_layer>/vars/0 (kernel) và /vars/1 (bias)
    write_txt_c(in, '/layers/conv2d/vars/0',   'conv0_W', '/layers/conv2d/vars/1',   'conv0_b', 0);
    write_txt_c(in, '/layers/conv2d_1/vars/0', 'conv1_W', '/layers/conv2d_1/vars/1', 'conv1_b', 0);
    write_txt_c(in, '/layers/conv2d_2/vars/0', 'conv2_W', '/layers/conv2d_2/vars/1', 'conv2_b', 0);
    write_txt_c(in, '/layers/conv2d_3/vars/0', 'conv3_W', '/layers/conv2d_3/vars/1', 'conv3_b', 0);
    write_txt_c(in, '/layers/conv2d_4/vars/0', 'conv4_W', '/layers/conv2d_4/vars/1', 'conv4_b', 0);
    write_txt_c(in, '/layers/conv2d_5/vars/0', 'conv5_W', '/layers/conv2d_5/vars/1', 'conv5_b', 0);
    write_txt_c(in, '/layers/conv2d_6/vars/0', 'conv6_W', '/layers/conv2d_6/vars/1', 'conv6_b', 0);
else
    % Lớp Fully-Connected
    write_txt_c(in, '/layers/dense/vars/0',   'dense1_W', '/layers/dense/vars/1',   'dense1_b', 1);
    write_txt_c(in, '/layers/dense_1/vars/0', 'dense2_W', '/layers/dense_1/vars/1', 'dense2_b', 1);
end
a=1;
end