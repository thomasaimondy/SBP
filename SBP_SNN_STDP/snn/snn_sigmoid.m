%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Feedforward propagation of SNN
%%% Input: layers, opts, mode
%%% mode 1 -- Norm Sigmoid (-1/1)
%%% mode 2 -- Hardsigmoid (0/1)
%%% mode 3 -- Spike (0/1)
%%% mode 4 -- Tanh (-1/1)
%%% mode 5 == ReLU (0/N)
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layers=snn_sigmoid(layers,opts,mode)
    
    % Sigmoid
    if(mode ==1)    
        layers = 2./(1+exp(-layers))-1;
    end
    
    % Hardsigmoid
    if(mode ==2)
        layers(layers>1)=1;
        layers(layers<0)=0.01*layers(layers<0);
    end
    
    % Spike
    if(mode == 3)
        layers = layers > opts.V_th;
    end
    
    % Tanh
    if (mode == 4)
        layers = tanh(layers);
    end
    
    % ReLU
    if(mode ==5)
        layers(layers<0)=0;
    end
    % none
    if(mode ==6)
        layers(layers<0)=0;
    end
    
end