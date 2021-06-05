

clearvars;
load('guesture_train_test_x_theta_strength_200.mat');
load('guesture_label.mat');
% normalizaton in x
[samples,dim] = size(train_x);
max_s = max(train_x(:,1:100),[],2);
max_t = max(train_x(:,100:end),[],2);
for i=1:1:samples
    train_x(i,1:100) = train_x(i,1:100)./max_s(i);
    train_x(i,100:end) = train_x(i,100:end)./max_t(i);
end

figure;
title('temporal');
grid on;
count = 1;
num = 3;
for i=1:1:samples
    if count>num*num
        break;
    end
    [~,label] = max(train_y(i,:));
    if label~=2 & label~=3
        continue;
    end
    subplot(num,num,count);
    plot(train_x(i,:));
    title(num2str(label));
    count = count +1;
end

% tempral features
figure;
title('features');
grid on;
count = 1;
for i=1:1:samples
    if count>num*num
        break;
    end
    [~,label] = max(train_y(i,:));
    if label~=2 & label~=3
        continue;
    end
    subplot(num,num,count);
    temporal = train_x(i,:);
    %features = msf_mfcc(temporal,16000,'nfilt',40,'ncep',12);
    features = msf_lpcc(temporal,16000,'order',10);
    
    plot(features);
    title(num2str(label));
    count = count +1;
end






%{
figure;
grid on;
title('frequency');
for i=1:1:samples
    if i>9
        break;
    end
    subplot(3,3,i);
    Y = fft(train_x(i,:),16384);%进行16384个点的fft变换
    Pyy = Y.* conj(Y) /16384;   %功率谱转换
    %f = 500*(0:8192)/16384;
    plot(Pyy);
    [~,label] = max(train_y(i,:));
    title(num2str(label));
end

%}

