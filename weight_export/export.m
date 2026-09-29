clear all;
rehash;

file_hdf5 = 'model.weights.h5';

% Xuất trọng số các lớp Convolution
a = write_weight_vgg_face_keras(file_hdf5, 0);

% Xuất trọng số các lớp Dense
b = write_weight_vgg_face_keras(file_hdf5, 1);

disp('==> THÀNH CÔNG! Đã xuất toàn bộ file .txt!');