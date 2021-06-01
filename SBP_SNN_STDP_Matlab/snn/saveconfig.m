%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Save all configurations of SNN
%%% Input: opts
%%% Output: Save as text
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ output_args ] = saveconfig( opts )
    % Get file name
    names = fieldnames(opts);
    % Save all parser variables
    for i=1:length(names)
        % Name 
        key = names{i};
        % Configuration
        value = getfield(opts, key);
        % Print values
        fprintf(opts.fid,'%s\t=\t%s\n',key,num2str(value));
    end
end

