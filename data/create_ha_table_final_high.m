function [ha_table,his_table] = create_ha_table_final_high(src_img,tag_img)
    src_img = uint16(src_img);
    tag_img = uint16(tag_img);
    ha_table = zeros(3,65536);
    his_table = zeros(3,65536);
    for c=1:1:3
        src_img_sub = src_img(:,:,c);
        tag_img_sub = tag_img(:,:,c);
        [src_his_sub,~] = imhist(src_img_sub, 65536);
        [tag_his_sub,~] = imhist(tag_img_sub, 65536);
        ha_table(c,:) = create_ha_table_sub(src_his_sub,tag_his_sub);
        his_table(c,:) = src_his_sub;
    end
end

function table = create_ha_table_sub(src_his, tag_his)
    table = zeros(1,65536);
    His3 = zeros(65536,1);
    His_End = 1; %%%starting position of the matching
for ii=1:65536
    His3(ii) = tag_his(ii);
end
for ii=1:65536
    if src_his(ii)==0 %%%empty bin
        table(ii) = -1;
    else %%%non-empty bin
        Pix_Sum = src_his(ii);
        Vua_Sum = 0;   
        flag = 1;
        jj = His_End;
        while(flag==1)           
            if His3(jj)<Pix_Sum
                Pix_Sum = Pix_Sum-His3(jj);
                Vua_Sum = Vua_Sum+His3(jj)*jj;
                jj = jj+1;
            else
                His_End = jj;
                His3(jj) = His3(jj)-Pix_Sum;
                Vua_Sum = Vua_Sum+Pix_Sum*jj;
                table(ii) = round(Vua_Sum/src_his(ii)); %%%averaging over all matched bins
                flag = 0;
            end
        end
    end
end
    table = table-1;
end