function a = write_txt_c( in,weight,file_name_w,bias,file_name_b,fc )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if(fc == 0)
   % kernel_1_0 = flip(flip(int16(reshape_arr(h5read(in,weight))*2^15)),2);
    kernel_1_0 = int16(reshape_arr(h5read(in,weight))*2^15);
    bias_1_0 = int16(h5read(in,bias)*2^15);
else
    kernel_1_0 = h5read(in,weight);
    bias_1_0 = h5read(in,bias);
end

file_name_w_t = strcat('C:/Users/DEL/Downloads/DATN/industrial_defect_30_7_2023_new/',file_name_w,'.txt');
file_name_b_t = strcat('C:/Users/DEL/Downloads/DATN/industrial_defect_30_7_2023_new/',file_name_b,'.txt');

fid1 = fopen( file_name_w_t, 'w' );
fid2 = fopen( file_name_b_t, 'w' );
%fprintf()
if(fc == 0)
     %fprintf(fid1,'%d %d %d %d\n',size(kernel_1_0));
     %fprintf(fid2,'%d\n',size(bias_1_0,1));
    for i=1:32:size(kernel_1_0,4)
        for j=1:size(kernel_1_0,3)
            for m=1:3
            %fprintf(fid1,'%f %f %f\n',kernel_1_0(1,1,j,i),kernel_1_0(1,2,j,i),kernel_1_0(1,3,j,i));
            %fprintf(fid1,'%f %f %f\n',kernel_1_0(2,1,j,i),kernel_1_0(2,2,j,i),kernel_1_0(2,3,j,i));
            %fprintf(fid1,'%f %f %f\n',kernel_1_0(3,1,j,i),kernel_1_0(3,2,j,i),kernel_1_0(3,3,j,i));
            % flip
            %fprintf(fid1,'%d %d\n',kernel_1_0(3,3,j,i),kernel_1_0(3,2,j,i));
            %fprintf(fid1,'%d %d\n',kernel_1_0(3,1,j,i),kernel_1_0(2,3,j,i));
            %fprintf(fid1,'%d %d\n',kernel_1_0(2,2,j,i),kernel_1_0(2,1,j,i));
            %fprintf(fid1,'%d %d\n',kernel_1_0(1,3,j,i),kernel_1_0(1,2,j,i));
               for n=1:3
                   if((m == 3) && (n == 3))
                       break;
                   end
                    for t=1:16
                        fprintf(fid1,'%d ',typecast(int16(kernel_1_0(m,n,j,t+i-1)),'uint16'));
                       % a=typecast(int16(kernel_1_0(m,n,j,t+i-1)),'uint16')
                        if(mod(t,2) == 0)
                            fprintf(fid1,'\n');
                        end
                    end
                    
                    for t=17:32
                        fprintf(fid1,'%d ',typecast(int16(kernel_1_0(m,n,j,t+i-1)),'uint16'));
                        if(mod(t,2) == 0)
                            fprintf(fid1,'\n');
                        end
                    end
               end                 
            end
            
            
        end
        for j=1:size(kernel_1_0,3)
                        for t=1:16
                            fprintf(fid1,'%d ',typecast(int16(kernel_1_0(3,3,j,t+i-1)),'uint16'));
                           % a=typecast(int16(kernel_1_0(m,n,j,t+i-1)),'uint16')
                            if(mod(t,2) == 0)
                                fprintf(fid1,'\n');
                            end
                        end
                    
                        for t=17:32
                            fprintf(fid1,'%d ',typecast(int16(kernel_1_0(3,3,j,t+i-1)),'uint16'));
                            if(mod(t,2) == 0)
                                fprintf(fid1,'\n');
                            end
                        end
                
            end
    end


    for i=1:16:size(bias_1_0,1)
        for j=1:16
            fprintf(fid2,'%d ',typecast(int16(bias_1_0(j+i-1)),'uint16'));
            if(mod(j,2) == 0)
                fprintf(fid2,'\n');
            end
        end
    end

    fclose(fid1);
    fclose(fid2);
else
    %fprintf(fid1,'%d %d\n',size(kernel_1_0));
    %fprintf(fid2,'%d\n',size(bias_1_0,1));
    
    for i=1:size(kernel_1_0,1)
        for j=1:size(kernel_1_0,2)
                    fprintf(fid1,'%f\n',kernel_1_0(i,j));
                  % if(j == size(kernel_1_0,2)) 
                  %      fprintf(fid1,'\n');
                  % end
        end
    end
    
    for i=1:size(bias_1_0,1)
            fprintf(fid2,'%f\n',bias_1_0(i));
    end
    
    fclose(fid1);
    fclose(fid2);
end
a=1;
end

