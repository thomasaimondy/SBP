  clear;clc;
    currentFolder = pwd;
    addpath(genpath(currentFolder))
    
    tic;
    rand('seed',100);
    load('DVS_gesture_100.mat');
    load('gesture_label.mat');
    xsize = size(train_x);
    [~,expected]=max(train_y,[],2);
   
% print a full image
 fullimglen = 25;
 fullimg = zeros(32*11, 32*fullimglen);

    for i=1:1:xsize(1)
        mkdir(['showimgs/' num2str(i) '_' num2str(expected(i))]);
        for j=1:1:xsize(2)
            im = reshape(train_x(i,j,:),32,32);
            imwrite(im',['showimgs/' num2str(i) '_' num2str(expected(i)) '/' num2str(j) '.jpeg']);
            if j<=fullimglen
                fullimg((expected(i)-1)*32+1:expected(i)*32,(j-1)*32+1:j*32)=im';
            end
        end
        if i>=12
            break;
        end
    end
    toc;
    imwrite(fullimg,'full_img.png');
    

 