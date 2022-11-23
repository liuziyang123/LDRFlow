function full_table = table_interpolate(src_table, interp_table, inter)
    lin_tensor = 1:1:65536;
    full_table = zeros(3,65536);
    for c=1:1:3
        index_list = uint8(index_search(src_table(c,:)));
        index_num = sum(index_list(:))+1;
        if interp_table(c, index_num+inter) == 65535
            interp_table(c, index_num+inter)
        else
            f = (interp_table(c, index_num+inter) - interp_table(c, index_num)) / inter * (lin_tensor - index_num) + interp_table(c, index_num);
            full_table(c, :) = single(interp_table(c, :)).*single(1-index_list) + single(index_list).*single(f);
        end
    end
    full_table = round(full_table);
    full_table(full_table<0) = 0;
    full_table(full_table>65535) = 65535;
end