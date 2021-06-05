function [ output_args ] = tnethomas( input_args )
%TNETHOMAS 此处显示有关此函数的摘要
%   此处显示详细说明

% Load data
load layer2_all
load mnist_test
for i =1:1:5
    train_x = layer2_all{i};
    train_labels=test_labels;
    
    % Set parameters
    no_dims = 2;%压缩后的维度
    initial_dims = 50;%PCA降维（在运行 tsne 函数之前，会自动使用 PCA 对数据预处理，将原始样本集的维度降低）
    perplexity = 30; %高斯分布的perplexity，越是高密度的样本，其值越大，一般推荐5-50
    % Run t-SNE
    mappedX = tsne(train_x, [], no_dims, initial_dims, perplexity);
    % Plot results
    gscatter(mappedX(:,1), mappedX(:,2),train_labels); %train_labels生成不同的标签颜色
end

end

