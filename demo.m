%% Demo for unmixing required by a hyperspectral scene simulator
% with KMSCD (k-means sparse coding dictionary) learning algorithm and representation estimation with FNNOMP (fast non-negative orthogonal matching pursuit) for abundance estimation with desired sparsity constraint

% Notes
% 1. Generate a lookup table (LUT) from the 'abundance' and the generated database of 'endmembers' as your scene simulator file demands.
% 2. Expect endmembers to have high values in the presence of outliers and over-illuminated parts of the scene (refer lines 13-15 of estFNNOMPabundance.m file)
% 3. Uncomment line 148 of KMSCDUnmix.m file if you wish to force all endmembers to have positive values (in theory endmembers should all be positive but noise, atmospheric compensation method error, and other factors may force negative values)

function demo

% demo with modesto scene scaled down. the image has been scaled down to reduce storage space in this repository demo.
load('modesto_resize.mat');

% Run the algorithm, and pre-process as needed like band selection and de-noising algorithms
% demo run with maximum 30 endmembers and 8,000 iterations with a maximum of 4 materials per pixel (3 endmember materials, and a zeros vector for shade/illumination). Change as necessary.
endmembers = KMSCDUnmix(reflectance, 30, 4, 8000);
[endmembers, abundance] = estFNNOMPabundance(endmembers, reflectance, 4);
clc;
save('output', 'endmembers', 'abundance');

end
