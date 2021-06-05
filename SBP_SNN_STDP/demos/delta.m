%t是一个全零的向量
%tsp表示的是这个向量的第tsp 第2*tsp 3*tsp... 地方为1
function t = delta(t,tsp)
    num = ceil(length(t)/tsp);
    for i = 1:num
        t(i*tsp) = 1;
    end

end