function feat_train_x = convertFrames2one(train_x)
% conver the raw data with 1176,100,1024 dimensions into 1176,1,1024
% dimensions, This method will be failed!!!!

%if ~exist('guesture_train_test_x_theta_strength_200.mat','file')
%                train_x = convertFrames2one(train_x);
%                test_x = convertFrames2one(test_x);
%                save('guesture_train_test_x_theta_strength_200.mat','train_x','test_x');
%            else
%                load('guesture_train_test_x_theta_strength_200.mat');
%end
            

imgw = 32;
imgh = 32;
[samnum,framnum,imgdim] = size(train_x);
new_train_x = zeros(samnum,framnum*2);
for i=1:1:samnum
    i
    new_img = zeros(1,framnum*2); % 1:framnum for strength, framnum:2*framnum for theata
    for j=1:1:framnum
        img = reshape(train_x(i,j,:),32,32);
        [ center ,img_processed ] = getCenters( img );
        zerop = [imgw/2,imgh/2];
        % strength
        strength = sqrt((center(2) - zerop(2))^2 + (center(1) - zerop(1))^2);
        new_img(j) = strength;
        % angle
        diff = (center(1) - zerop(1)) / strength;
        theta = acos(diff);
        new_img(framnum+j) = theta;
        
    end
    new_train_x(i,:) = new_img;
end
new_train_x(isnan(new_train_x)) = 0;

% normalizaton in new_train_x
[samples,dim] = size(new_train_x);
max_s = max(new_train_x(:,1:100),[],2);
max_t = max(new_train_x(:,100:end),[],2);
for i=1:1:samples
    new_train_x(i,1:100) = new_train_x(i,1:100)./max_s(i);
    new_train_x(i,100:end) = new_train_x(i,100:end)./max_t(i);
end

% convert from temporal info to sequential features
% features = msf_mfcc(temporal,16000,'nfilt',40,'ncep',12);
feat_train_x = zeros(samnum,20); % 10 for msf_lpcc
for i=1:1:samples
    temporal = new_train_x(i,:);
    features = msf_lpcc(temporal,16000,'order',20);%16000
    feat_train_x(i,:) = features;
end

end