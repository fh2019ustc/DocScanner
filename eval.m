path_rec = "xxx";  % rectified image path
path_scan = './scan/';  % scan image path
label_path = './layout/'; % layout result path

tarea = 598400;
ms1 = 0;
ld1 = 0;
lid1 = 0;
ms2 = 0;
ld2 = 0;
lid2 = 0;
wv = 0;
wh = 0;

sprintf(path_rec)
for i=1:65
    path_rec_1 = sprintf("%s%d%s", path_rec, i, '_1 copy_rec.png');  % rectified image path
    path_rec_2 = sprintf("%s%d%s", path_rec, i, '_2 copy_rec.png');  % rectified image path
    path_scan_new = sprintf("%s%d%s", path_scan, i, '.png');  % corresponding scan image path
    bbox_i_path = sprintf("%s%d%s", label_path, i, '.txt');   % corresponding layout txt path
    
    % imread and rgb2gray
    A1 = imread(path_rec_1);
    A2 = imread(path_rec_2);

%    if i == 64
%        A1 = rot90(A1,-2);
%        A2 = rot90(A2,-2);
%    end

    ref = imread(path_scan_new);
    A1 = rgb2gray(A1);
    A2 = rgb2gray(A2);
    ref = rgb2gray(ref);
    bbox_i = read_txt(bbox_i_path);
    bbox_i = bbox_i + 1; % python index starts from 0

    % resize
    b = sqrt(tarea/size(ref,1)/size(ref,2));
    ref = imresize(ref,b);
    A1 = imresize(A1,[size(ref,1),size(ref,2)]);
    A2 = imresize(A2,[size(ref,1),size(ref,2)]);
    scaled_bbox_i = bbox_i * b * 0.5;
    scaled_bbox_i = round(scaled_bbox_i);
    scaled_bbox_i = max(scaled_bbox_i, 1);

    % calculate
    [ms_1, ld_1, lid_1, W_v_1, W_h_1] = evalUnwarp(A1, ref, scaled_bbox_i);
    [ms_2, ld_2, lid_2, W_v_2, W_h_2] = evalUnwarp(A2, ref, scaled_bbox_i);
    ms1 = ms1 + ms_1;
    ms2 = ms2 + ms_2;
    ld1 = ld1 + ld_1;
    ld2 = ld2 + ld_2;
    lid1 = lid1 + lid_1;
    lid2 = lid2 + lid_2;
    wv = wv + W_v_1 + W_v_2;
    wh = wh + W_h_1 + W_h_2;
end

ms = (ms1 + ms2) / 130  % MS-SSIM
ld = (ld1 + ld2) / 130  % local distortion
li_d = (lid1 + lid2) / 130  % line distortion
wv = wv / 130  % wv index
wh = wh / 130  % wh index
