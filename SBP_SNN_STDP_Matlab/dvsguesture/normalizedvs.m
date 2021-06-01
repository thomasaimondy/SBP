function [ data_x_new, data_y_new ] = normalizedvs( data_x, data_y )
%NORMALIZEDVS 

[samples,frames,dimx] = size(data_x);
%data_x_new = squeeze(mean(data_x(:,1:50,:),2));
data_x_new = zeros(samples*frames,dimx);

[~,dimy] = size(data_y);
data_y_new = zeros(samples*frames,dimy);
for s=1:1:samples
    for f=1:1:frames
        data_x_new((s-1)*f+f,:) = data_x(s,f,:);
        data_y_new((s-1)*f+f,:) = data_y(s,:);
    end
end

end

