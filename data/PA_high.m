function output =  PA_high(map,input)
    img = input;
    [h,w,~] = size(img);
for c = 1:3
    for i = 1:h
       for j = 1:w
            img(i,j,c) = map(c,input(i,j,c)+1);
       end
    end
end
    output=uint16(img);
end