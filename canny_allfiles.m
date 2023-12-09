folder_name="D:\Knee";
views  = ["coronal","sagittal"];
splits = ["train","val","test"];
conditions = ["normal","torn"];
for view = views
    for split = splits
        for condition = conditions
            
            base_folder = sprintf("%s_%s_acl-split\\%s\\%s",folder_name,view,split,condition);
            new_folder = sprintf("%s_%s_acl-split-canny\\%s\\%s",folder_name,view,split,condition);
            mkdir(new_folder)

            files = dir(sprintf('%s\\*.png',base_folder));
            names = vertcat(files.name);
            
%             for i = 1:numel(files)
%                 %fprintf('Found file: %s\n', files(i).name);
%                 im=imread(sprintf('%s\\%s',base_folder,names(i,:)));
% 
%                 im_canny=canny_edge(im);
%                 imwrite(im_canny,sprintf('%s\\%s',new_folder,names(i,:)))
%                 if(mod(i,1000)==0)
%                     fprintf("\n%d images completed in %s %s %s set",i,view,split,condition)
%                 end
%             end
        end
    end
end


