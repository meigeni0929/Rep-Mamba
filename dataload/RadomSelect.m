%������Ҫ�ǽ������ݼ���ѡȡ
%�Ƚ������ݼ���ѡȡ��Ȼ��������ݵ��ϲ�������
clc;close all;clear;

%AID_dir = 'E:\Desktop\ymr\publication\AID\AID_dataset\AID\';  %ԭ���ݵ�λ��
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
 %   select_rule = 100 + 30;   %ѡ���ͼƬ����
%    rand_num = randperm(img_num, select_rule);  %ÿ������£�100������ѵ����30�����ڲ���

    for i=1:1:(select_rule)
        idx = rand_num(i);
        if i<=100
            img_save_folder = 'E:\Desktop\ymr\publication\AID\train\GT2\';  %�����ѵ������ѵ����������
        else
            img_save_folder = ['E:\Desktop\ymr\publication\AID\test\AID900\GT2\',class_folder,'\']; %����Ĳ��Լ������Լ�Ҫ����
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