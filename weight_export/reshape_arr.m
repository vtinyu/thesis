function out_arr = reshape_arr( in_arr )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

size_arr = size (in_arr);
for i=1:size_arr(4)
    for j=1:size_arr(3)
        for m=1:size_arr(2)
            for n=1:size_arr(1)
               out_arr(i,j,m,n) =  in_arr(n,m,j,i);
            end
        end
    end
end

end

