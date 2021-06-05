
function [mappedX_train, mappedX_test] = guesture_layers_tsne(train_x,test_x)
% this method will be failed.!!!
% use the tsne to lower 100*1024 to 1024 dimension
%  if ~exist('guesture_train_test_x_tsne.mat','file')
%    [train_x,test_x] = guesture_layers_states(train_x,test_x);
%    save('guesture_train_test_x_tsne','train_x','test_x');
%else
%    load('guesture_train_test_x_tsne.mat');
% end       
            
% the neighbour-hood layers states
totalx = [train_x;test_x];
task = 'guesture';
layer = 'layer'; %
plasticity = 'sbp';
load('DVS_gesture_100.mat'); % codes in guesture folder
load('guesture_label.mat');


tsne_x = reshape(totalx,size(totalx,1),size(totalx,2)*size(totalx,3)); % tsne_x with dimonsion NxD dataset X
initial_dims = size(train_x,3);% PCA initional processing dimonsions
no_dims =  500;% output dimension
perplexity = 50; % perplexity of gaussion processing, higher values for denser dataset,10-50
mappedX = tsne(tsne_x, [], no_dims, initial_dims, perplexity);

mappedX_train = mappedX(1:size(train_x,1),:);
mappedX_test = mappedX(size(train_x,1)+1:end,:);

% mappedX_train with dimension 1176, 1024
% mappedX_train with dimension 288, 1024
end