

load('sv_nn_guesture_sbp.mat');     % load nn
load('DVS_gesture_100.mat');        % codes in guesture folder
load('gesture_label.mat');         
train_x = normalizedvs(train_x);
test_x = normalizedvs(test_x);
[layer1, layer2,layer3] = snntest(nn,test_x,test_y);
[~,y_predict]=max(layer3,[],2);
[~,expected]=max(test_y,[],2);
bad=find(y_predict~=expected);
error=numel(bad)/size(test_y,1);

showlist = zeros(1,11);
for i=1:1:size(bad)
    classid = expected(bad(i)); % from 1 to 11
    showlist(classid) = showlist(classid) + 1;
end

figure;
bar(showlist);
