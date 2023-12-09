function im_edge = canny_edge(im)
    filt_im = imgaussfilt(im,1);
    im_edge = im2uint8(edge(im,'canny',[0.09, 0.11]));
end