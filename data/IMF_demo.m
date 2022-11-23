src_img  = imread('img2.png');
tag_img  = imread('img3.png');

table = create_ha_table_final_high(src_img, tag_img);
full_table = table_interpolate(table, full_table, 30);

test_img = PA_high(table, src_img);

imwrite(test_img, 'img2-img3.png');