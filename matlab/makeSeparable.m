function output = makeSeparable(input)
% output = makeSeparable(input)
%   Mean subtraction to account for -1/1 mask --> 0/1 mask
%		Refer to Asif, et al (2017)

rowMeans = mean(input,2);
colMeans = mean(input,1);
allMean = mean(rowMeans,1);

output = bsxfun(@minus, input, rowMeans);
output = bsxfun(@minus, output, colMeans);
output = bsxfun(@plus, output, allMean);
