 function [train_x, train_y, test_x, test_y] = load_dataset()
 
 rand('state', 0);
 load mnist_uint8;
 train_x = double(train_x) / 255;
 test_x  = double(test_x)  / 255;
 train_y = double(train_y);
 test_y  = double(test_y);
 
 
 %train_x(train_x>=0.01)=1;
 %train_x(train_x<0.01)=0;
 
 %test_x(test_x>=0.01)=1;
 %test_x(test_x<0.01)=0;
 end

