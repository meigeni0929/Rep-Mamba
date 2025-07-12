%这里主要是进行数据集的选取
%先进行数据集的选取，然后进行数据的上采样处理
clc;close all;clear;

%AID_dir = 'E:\Desktop\ymr\publication\AID\AID_dataset\AID\';  %原数据的位置
%AID_class_name = {'Airport\','BareLand\','BaseballField\','Beach\',...
%    'Bridge\','Center\','Church\','Commercial\','DenseResidential\','Desert\','Farmland\',...
%    'Forest\','Industrial\','Meadow\','MediumResidential\','Mountain\','Park\','Parking\',...
%    'Playground\','Pond\','Port\','RailwayStation\','Resort\','River\','School\','SparseResidential\',...
%    'Square\','Stadium\','StorageTanks\','Viaduct\'};
%length(AID_class_name)
%for n=1:1:length(AID_class_name)
%    class_folder =  AID_class_name{n};
%    jpg_list = dir(fullfile(AID_dir,class_folder,'*.jpg'));
%    img_num = length(jpg_list);
%     training_num = img_num/2;
%     test_num = 10;
 %   select_rule = 100 + 30;   %选择的图片数量
%    rand_num = randperm(img_num, select_rule);  %每个类别下，100的用于训练，30张用于测试

    for i=1:1:(select_rule)
        idx = rand_num(i);
        if i<=100
            img_save_folder = 'E:\Desktop\ymr\publication\AID\train\GT2\';  %保存的训练集，训练集不分类
        else
            img_save_folder = ['E:\Desktop\ymr\publication\AID\test\AID900\GT2\',class_folder,'\']; %保存的测试集，测试集要分类
        end

        if ~exist(img_save_folder,'dir')
            mkdir(img_save_folder);
        end

        img = imread([AID_dir,class_folder,jpg_list(idx).name]);
        img = im2double(img);
        img = img(44:555, 44:555, :);
%         img = imresize(img, [512, 512], 'bicubic');
        png_name = replace(jpg_list(idx).name,'jpg','png');
        imwrite(img,[img_save_folder,png_name]);
    end
end