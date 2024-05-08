function [ms, ld, li_d, wv, wh] = evalUnwarp(A, ref, data)
%EVALUNWARP compute MSSSIM and LD between the unwarped image and the scan
%   A:      unwarped image
%   ref:    reference image, the scan image
%   ms:     returned MS-SSIM value
%   ld:     returned local distortion value
%   Matlab image processing toolbox is necessary to compute ssim. The weights 
%   for multi-scale ssim is directly adopted from:
%
%   Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural 
%   similarity for image quality assessment." In Signals, Systems and Computers, 
%   2004. Conference Record of the Thirty-Seventh Asilomar Conference on, 2003. 
%
%   Local distortion relies on the paper:
%   Liu, Ce, Jenny Yuen, and Antonio Torralba. "Sift flow: Dense correspondence 
%   across scenes and its applications." In PAMI, 2010.
%
%   and its implementation:
%   https://people.csail.mit.edu/celiu/SIFTflow/

x = A;
y = ref;

im1=imresize(imfilter(y,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
im2=imresize(imfilter(x,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');

im1=im2double(im1);
im2=im2double(im2);

cellsize=3;
gridspacing=1;

sift1 = mexDenseSIFT(im1,cellsize,gridspacing);
sift2 = mexDenseSIFT(im2,cellsize,gridspacing);

SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;


[vx,vy,~]=SIFTflowc2f(sift1,sift2,SIFTflowpara);

rows1p = size(im1,1);
cols1p = size(im1,2);

% Li-D
rowstd_sum = 0;
for i = 1:rows1p
    rowstd = std(vy(i, :),1); 
    rowstd_sum = rowstd_sum + rowstd;
end
rowstd_mean = rowstd_sum / rows1p;

colstd_sum = 0;
for i = 1:cols1p
    colstd = std(vx(:, i),1); 
    colstd_sum = colstd_sum + colstd;
end 
colstd_mean = colstd_sum / cols1p;
    
li_d = (rowstd_mean + colstd_mean) / 2;


% LD
d = sqrt(vx.^2 + vy.^2);
ld = mean(d(:));


% MS-SSIM
wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
ss = zeros(5, 1);
for s = 1 : 5
    ss(s) = ssim(x, y);
    x = impyramid(x, 'reduce');
    y = impyramid(y, 'reduce');
end
ms = wt * ss;


% wv and wh
rowstd_sum = 0;
for i = 1:size(data, 1)
    rowstd_top = std(vy(data(i,2), data(i,1):data(i,3)),1) / (data(i,3)-data(i,1)); 
    rowstd_bot = std(vy(data(i,4), data(i,1):data(i,3)),1) / (data(i,3)-data(i,1)); 
    rowstd_sum = rowstd_sum + rowstd_top + rowstd_bot;
end
wv = rowstd_sum / (2 * size(data, 1));

colstd_sum = 0;
for i = 1:size(data, 1)
    colstd_left = std(vx(data(i,2):data(i,4), data(i,1)),1) / (data(i,4)- data(i,2)); 
    colstd_right = std(vx(data(i,2):data(i,4), data(i,3)),1) / (data(i,4)- data(i,2)); 
    colstd_sum = colstd_sum + colstd_left + colstd_right;
end 
wh = colstd_sum / (2 * size(data, 1));

end
