function full_table = index_search(src_table)
    full_table = zeros(1, 65536);
    for c=1:1:65536
        if src_table(c)>=0
            break;
        else
            full_table(1, c) = 1;
        end
    end
end