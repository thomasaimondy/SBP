
% the neighbour-hood layers states
totalx = [train_x;test_x];
task = 'guesture';
layer = 'layer'; %
plasticity = 'sbp';
%load('DVS_gesture_100.mat'); % codes in guesture folder
%load('guesture_label.mat');
%load('guesture_train_test_x_theta_strength_200.mat');
load('dvs_normalization.mat');

tsne_x = train_x; % tsne_x with dimonsion NxD dataset X
tsne_y = train_y;

initial_dims = round(size(tsne_x,1)/2);% PCA initional processing dimonsions
no_dims =  2;% output dimension
perplexity = 30; % perplexity of gaussion processing, higher values for denser dataset,10-50
mappedX = tsne(tsne_x, [], no_dims, initial_dims, perplexity);
tsne_labels=genLabels(tsne_y); % with dimonsion 

h = figure;
gscatter(mappedX(:,1), mappedX(:,2),tsne_labels); %train_labels generate colors
savefig(h,['analysis/' task,'_',plasticity '_' layer '.fig']);
save( ['analysis/' task,'_',plasticity '_' layer '.mat'],'mappedX','tsne_labels');
%mappedX_train = mappedX(1:size(train_x,1),:);
%mappedX_test = mappedX(size(train_x,1)+1:end,:);