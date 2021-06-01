function [ center ,img ] = getCenters( img )
% get the center of most connected are of image

L = bwlabel(img); % label the connected region
stats = regionprops(L);
Ar = cat(1, stats.Area);
ind = find(Ar ==max(Ar)); % find the max connection area id
try
   img(find(L~=ind(1)))=0;% set other regions as zero
catch exception
   center = [0,0]; % null frame
   disp('null frame, error');
   return;
end



[w,h] = size(img);
centx = [];
centy = [];
for i=1:1:w
    for j=1:1:h
        if img(i,j)>0
            centx = [centx i];
            centy = [centy j];
        end
    end
end
meanx = round(mean(centx));
meany = round(mean(centy));
center = [meanx,meany];
%figure,imshow(img);% show the big connected region
%imb = zeros(size(img));
%imb(meanx,meany)=1;
%figure,imshow(imb);% show the big connected region

end

