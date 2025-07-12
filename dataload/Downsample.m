clear;
close all;
% 数据的批量上（下）采样
%ans=1
AID_class_name = {'Airport\','BareLand\','BaseballField\','Beach\',...
    'Bridge\','Center\','Church\','Commercial\','DenseResidential\','Desert\','Farmland\',...
    'Forest\','Industrial\','Meadow\','MediumResidential\','Mountain\','Park\','Parking\',...
    'Playground\','Pond\','Port\','RailwayStation\','Resort\','River\','School\','SparseResidential\',...
    'Square\','Stadium\','StorageTanks\','Viaduct\'};
LR_clip_folder = 'G:\TTSA\TTST-main\AID-tiny\GT\';

save_LR_folder = 'G:\TTSA\TTST-main\AID-tiny\LR\';

save_upsample_folder = 'G:\TTSA\TTST-main\AID-tiny\LR\';
save_bicubic_folder = 'G:\TTSA\TTST-main\AID-tiny\Bicubic\';

if ~exist(save_upsample_folder,'dir')
    mkdir(save_upsample_folder);
end
if ~exist(save_bicubic_folder,'dir')
    mkdir(save_bicubic_folder);
end
%
%length(AID_class_name)
for i = 1:1:length(AID_class_name)
    class_folder =  AID_class_name{i};
    filepath = dir(fullfile(LR_clip_folder,class_folder,'*.png'));  %获取文件夹下所有文件扩展名为JPG的文件
    up_scale=4;
    filepath_=[save_LR_folder,class_folder]; %获取LR数据
    %filepath_=[save_bicubic_folder,class_folder];  %获取bicubic
    if ~exist(filepath_,'dir')
    mkdir(filepath_);
    end
    for j=1:1:length(filepath)
        img_name = filepath(j).name;
        img = imread(fullfile(LR_clip_folder,class_folder,filepath(j).name));%获取LR数据
        %img = imread(fullfile(save_LR_folder,class_folder,filepath(j).name)); %获取bicubic数据
        img = im2double(img);

        im_LR= imresize(img, 1/up_scale, 'bicubic');  %先进行下采样
        %im_bic = imresize(img, up_scale, 'bicubic');   %上采样数据
        img_name = filepath(j).name;
        imwrite(im_LR,fullfile(save_LR_folder,class_folder,img_name)); %保存LR数据
        %imwrite(im_bic,fullfile(save_bicubic_folder,class_folder,img_name)); %保存bicubic数据
    end 
end