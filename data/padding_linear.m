function full_table = padding_linear(src_table,empty_value)
    full_table = zeros(3,65536);
    for c=1:1:3
        range = 1:65536;
        sub_table = src_table(c,:);
        x = range(sub_table>empty_value);
        y = sub_table(sub_table>empty_value);
        full_table(c,:) = interp1(x,y,range,'linear','extrap');
    end
    full_table(full_table<0) = 0;
    full_table(full_table>65535) = 65535;
end