%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Information conversion: SNN Positive ReLU
%%% Input: x
%%% Output: Converted x
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = snn_positive_relu(x)
    % Initialization
    y = zeros(1,length(x));
    % Converting each item in x
    for i = 1:length(x)
        % Negative x makes -x
        % Positive x makes x
        if x(i)<=0
            y(i) = -x(i);
        else
            y(i) = x(i);
        end
    end
end