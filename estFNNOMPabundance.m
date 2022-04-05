function [endmembers, abundance] = estFNNOMPabundance(endmembers, image, maxMatsPerPixel)
clc;
[h, w, b] = size(image);
image = reshape(double(image), [h*w, b]);
image = image';
[nD, ~] = size(endmembers);
endmembers = double(endmembers');

abundance = single(zeros(h*w, nD));
for px = 1:h*w
    abundance(px, :) = single(fnnomp([endmembers; ones(1, nD)], [image(:, px); 1], maxMatsPerPixel-1, 0));
end
m = max(abundance(:));
abundance = abundance / m;
endmembers = endmembers * m; % this will result in a high value (reflectance > 1) endmembers in the presence of outliers

s = find(sum(abundance) == 0);
endmembers(:, s) = []; % removing endmemembers with zero abundance because they do not contribute to the reconstructed scene.
abundance(:, s) = [];
endmembers = [endmembers, zeros(size(endmembers, 1), 1)]; % add a zeros vector for shade/illumination
abundance = [abundance, zeros(size(abundance, 1), 1)];
abundance(:, end) = 1 - sum(abundance(:, 1:end-1), 2);
abundance = reshape(single(abundance), [h, w, size(endmembers, 2)]);
endmembers = endmembers';

end