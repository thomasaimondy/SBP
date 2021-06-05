%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Nonlinear for SBP, works for normalization
%%% Input: X
%%% Output: Normalized X
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ output ] = sbpnonlinear( input )
    if sum(input)==0
       % All-zero X, keep it
       output = input;
    else
       % Normalization with x/sum(x)
       output = input/(sum(input));
    end
end

