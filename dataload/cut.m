%处理数据
% 输入和输出文件夹路径
inputFolder = 'G:\TTSA\TTST-main\AID-tiny\GT\';
outputFolder = 'G:\TTSA\TTST-main\AID-tiny\output\';

% 获取输入文件夹下所有的jpg图像文件
jpgFiles = dir(fullfile(inputFolder, '*.jpg'));
for k = 1:length(jpgFiles)
    % 构建完整的输入和输出文件路径
    inputFileName = fullfile(inputFolder, jpgFiles(k).name);
    [~, baseFileName, ~] = fileparts(jpgFiles(k).name);
    outputFileName = fullfile(outputFolder, [baseFileName, '.png']);
    % 读取图像
    img = imread(inputFileName);
    img = im2double(img);
    img = img(44:555, 44:555, :);
%   img = imresize(img, [512, 512], 'bicubic');
    %png_name = replace(jpg_list(idx).name,'jpg','png');
    % 保存为PNG格式
    imwrite(img, outputFileName, 'png');
end
