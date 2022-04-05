%% KMSCD Endmember Learning with FNNOMP representation
% Published paper (MDPI): https://doi.org/10.3390/jimaging5110085
% C-SCD source code originally authored by Adam Charles and this code is modified by Ayan Chatterjee. Email: ayan@outlook.com. Please see relevant links.
% Note: Replace FNNOMP with Non-negative least squares function or other sparse representation as required. FNNOMP for representation is recommended to accomodate hyperspectral scene simulator constraints.

%% Relevant Links (Last accessed 29 September 2019)
% SCD-SOMP Paper (IEEE LOCS): https://doi.org/10.1109/LOCS.2019.2938446
% C-SCD Paper (IEEE JSTP): https://doi.org/10.1109/JSTSP.2011.2149497
% C-SCD code: http://adamsc.mycpanel.princeton.edu/documents/Dictionary_Learning_Library_v1-0.zip
% FNNOMP Paper (IEEE SPL): https://doi.org/10.1109/LSP.2015.2393637
% FNNOMP code: http://www.mehrdadya.com/code/NNOMPv1.0.tar.gz

%% Inputs
% Image -> 3D hyperspectral or multispectral cube
% num_endmembers -> number of atoms to learn
% maxMatsPerPixel - > maximum number of endmember materials per pixel
% maxIter -> maximum number of iterations

function endmembers = KMSCDUnmix(Image, num_endmembers, maxMatsPerPixel, maxIter)

clc;
[h, w, nB] = size(Image);
Image = reshape(single(Image), [h*w, nB]);

opts.save_name = 'temp_1.mat';      % Save Name in case of errors
opts.grad_type = 'norm';            % Choose weather to include the Forb norm in E(a,D)
opts.n_elem = num_endmembers;       % Number of endmember materials
opts.max_sparsity = maxMatsPerPixel - 1; % maximum number of endmember materials per pixel. One endmember reserved for illumination changes.
opts.iters = maxIter;               % Number of learning iterations
opts.in_iter = 200;                 % Number of internal iterations
opts.GD_iters = 1;                  % Basis Gradient Descent iterations
opts.step_size = 0.01;              % Initial Step Size for Gradient Descent
opts.decay = 0.9998;                % Step size decay factor
opts.lambda = 0.5;                  % Lambda Value for Sparsity
opts.verb = 1;                      % Default to no verbose output

dictionary_initial = abs(rand(nB, opts.n_elem)); %create a random initial dictionary
dictionary_initial = dictionary_initial./(ones(nB, 1)*sqrt(sum(dictionary_initial.^2, 1))); % Basis normalized for l2 norm
kclass = kmeans(Image, opts.in_iter); % unsupervised clustering with kmeans
endmembers = train_endmembers(Image', dictionary_initial, kclass, opts); %learn dictionary
endmembers = endmembers';

end

function endmembers_end = train_endmembers(data_obj, initial_dict, kclass, opts)
% OPTIONS: Make sure that the correct options are set and that all
% necessary variables are available or defaulted.

if ~isfield(opts, 'save_name')
    date_str = date;
    opts.save_name = [date_str(8:end), date_str(3:7), date_str(1:2), ...
        'Dictionary_' num2str(opts.n_elem), 'Elems_', num2str(opts.lambda), ...
        'lambda.mat'];
    fprintf('Save name not specified, saving as %s...\n', opts.save_name)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Error Checking
if opts.n_elem ~= size(initial_dict, 2)
    error('Dimension mismatch between opts.n_elem and initial dictionary size!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations and Dimention Extraction
% Initialize Basis
endmembers_n = initial_dict; 
% Iteration counter initialization
iter_num = 0;
% Initialize step size
step_s = opts.step_size;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run Algorithm
fprintf('Educating your basis...\n')
basic_cell.options = opts;
while iter_num < opts.iters
    try
        %% Get Training Data. Modified by Ayan Chatterjee.
        data_use_ind = zeros(1, opts.in_iter);
        parfor i = 1:opts.in_iter
            matpos = find(kclass == i);
            data_use_ind(i) = matpos(ceil(numel(matpos)*rand(1)));
        end
        x_im = data_obj(:, data_use_ind); % Initialize matricies that will be populated during the actual learning
        %% Interior loop: find sparse coefficients 
        coef_vals = gen_multi_infer(double(endmembers_n), double(x_im), opts);
        %% Minimize the energy w.r.t. the dictionary using gradient descent
       	endmembers_n = endmembers_update(x_im, endmembers_n, coef_vals, step_s, opts);
        iter_num = iter_num + 1; % update the iteration count
        if opts.verb == 1
            %Spit out info
            im_snr = mean(sum(x_im.^2, 1)./sum((x_im - endmembers_n*coef_vals).^2, 1));
            disp(strcat("Iter: ", num2str(iter_num),", SNR: ", num2str(im_snr),", step size is ", num2str(step_s)));
        end
        % Update the step size
        step_s = step_s*opts.decay;
    catch ME
        fprintf('Saving last dictionary before error...\n')
        basic_cell.dictionary = endmembers_n;
        basic_cell.iter = iter_num;
        eval(sprintf('save %s basic_cell;', opts.save_name));
        fprintf(ME.message)
        fprintf('The program failed. Your dictionary at the last iteration was saved.')
        rethrow(ME)
    end 
end
endmembers_end = endmembers_n;

end

function coef_vals = gen_multi_infer(endmembers_n, x_im, opts)
% Initialize coefficients. Modified by Ayan Chatterjee
coef_vals = zeros(opts.n_elem, opts.in_iter);
for index_in = 1:opts.in_iter
    coef_vals(:, index_in) = fnnomp([endmembers_n; ones(1, opts.n_elem)], [x_im(:,index_in); 1], opts.max_sparsity, 0);
end
end

function endmembers_new = endmembers_update(x_im, endmembers_old, coef_vals, step_s, opts)

% function endmembers_new = endmembers_update(x_im, dictionary_old, coef_vals,
% step_s, opts)
% 
% Takes a gradient step with respect to the sparsity inducing energy
% function.
% 
% Inputs:
%   x_im        - Data samples over which to average the gradient step
%   endmembers_old    - The previous dictionary (used to infer the coefficients)
%   coef_vals   - The inferred coefficients for x_im using endmembers_old
%   step_s      - The step size to take in the gradient direction
%   opts        - Options for the particular problem (outlined in
%                 learn_dictionary.m)
%
% Outputs:
%   endmembers_new    - The new dictionary after the gradient step
% 
% Last Modified 6/4/2010 - Adam Charles

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Take a gradient step
if strcmp(opts.grad_type, 'norm')
    for index2 = 1:opts.GD_iters
        % Take a step in the negative gradient of the basis:
        % Minimizing the energy:
        % E = ||x-Da||_2^2 + lambda*||a||_1^2
        % Update The basis matrix
        updateTerm = (x_im - endmembers_old*coef_vals)*coef_vals';
        endmembers_new = endmembers_old + step_s*updateTerm;
        % endmembers_new(endmembers_new < 0) = 0; % endmember reflectance cannot be less than 0
    end     
elseif strcmp(opts.grad_type, 'forb')
    for index2 = 1:opts.GD_iters
        % Take a step in the negative gradient of the basis:
        % This time the Forbenious norm is used to reduce unused
        % basis elements. The energy function being minimized is
        % then:
        % E = ||x-Da||_2^2 + lambda*||a||_1^2 + ||D||_F^2

        % Update The basis matrix
        endmembers_new = endmembers_old + (step_s)*((x_im - endmembers_old*coef_vals)*coef_vals'...
            - opts.lambda2*2*endmembers_old)*diag(1./(1+sum(coef_vals ~= 0, 2)));
    end  
end
end