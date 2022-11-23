clear;
clc;

input_dir = './Raw/Training';
output_dir = './IMF_short2long/Training';

flag = 0;
num = 0;
dir_list = dir(input_dir);
for i = 1:length(dir_list)
    if (isequal(dir_list(i).name, '.') || isequal(dir_list(i).name, '..') || ~dir_list(i).isdir)
        continue;
    end

    save_dir = fullfile(output_dir, dir_list(i).name);
    
    if ~exist(save_dir)
        mkdir(save_dir);
    end
    
    img_list = dir(fullfile(dir_list(i).folder, dir_list(i).name, '*.tif'));
    index_tag = floor(length(img_list)/2) + 1;
    for j =1:length(img_list)
        
        if j == index_tag
            continue
        end
        
        src_img = imread(fullfile(img_list(j).folder, img_list(j).name));
        tag_img = imread(fullfile(img_list(index_tag).folder, img_list(index_tag).name));
        src_name = img_list(j).name;
        tag_name = img_list(index_tag).name;
        
        % always normalize the darker image to the brighter one
        if mean(src_img(:)) > mean(tag_img(:))
            tmp = src_img;
            src_img = tag_img;
            tag_img = tmp;
            tmp = src_name;
            src_name = tag_name;
            tag_name = tmp;
        end
        
        table = create_ha_table_final_high(src_img, tag_img);
        test_img = PA_high(table, src_img);
        imwrite(tag_img, fullfile(save_dir, tag_name));
        imwrite(test_img, fullfile(save_dir, strrep(src_name, '.tif',  sprintf('-%s',tag_name))));
        T = padding_linear(table, -2);
        T = table_interpolate(table, T, 7000);
        filename = strrep(src_name, '.tif',  sprintf('-%s',tag_name));
        save(fullfile(save_dir, strrep(filename, 'tif', 'mat')), 'T');
        table = create_ha_table_final_high(tag_img, src_img);
        T = padding_linear(table, -2);
        T = table_interpolate(table, T, 30);
        filename = strrep(tag_name, '.tif',  sprintf('-%s',src_name));
        save(fullfile(save_dir, strrep(filename, 'tif', 'mat')), 'T');
        
    end
end