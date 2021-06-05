function [ color_labels ] = genLabels( input_y )
% generate labels for mnist and nettalk data.
% mnist, label : 1-10 for 10 classes
% nettalk, label: 1-26 for 16 output classes, however, 
%           output of nettalk may more than one classes, hence 
%           we give one sample one labels. 

[n,d] = size(input_y);
labelMap = containers.Map;
color = 1;
color_labels = zeros(n,1);
for i=1:1:n
    key = int2str(input_y(i,:));
    if ( isKey(labelMap,key)) % exist
        colors = values(labelMap, {key});
        color_labels(i) = colors{1};
        continue;
    else % first time
        labelMap(key) = color;
        color_labels(i) = color;
        color = color + 1;
    end
end

end

