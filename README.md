:boom: **Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 

# DocScanner
This is a PyTorch/GPU re-implementation of the paper [DocScanner: Robust Document Image Rectification with Progressive Learning](https://arxiv.org/pdf/2110.14968v2.pdf).

![image](https://user-images.githubusercontent.com/50725551/194188411-b0378999-2457-462b-97a9-fe85989ccae9.png)

## Evaluation
- ***Important.*** In the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html), the '64_1.png' and '64_2.png' distorted images are rotated by 180 degrees, which do not match the GT documents. It is ingored by most of existing works. Before the evaluation, please make a check. Note that the performances in most of existing work are computed with these two ***mistaken*** samples in [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html).
- For reproducing the following quantitative performance on the ***corrected*** [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html), please use the geometric rectified images available from [Google Drive](https://drive.google.com/drive/folders/1QBe26xJwIl38sWqK2ZE9ke5nu0Mpr4dW?usp=sharing). For the ***corrected*** performance of [other methods](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification), please refer to the paper [DocScanner](https://arxiv.org/pdf/2110.14968v2.pdf).
- ***Image Metrics:***  We use the same evaluation code for MS-SSIM and LD as [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset based on Matlab 2019a. Please compare the scores according to your Matlab version. We provide our Matlab interface file at ```$ROOT/ssim_ld_eval.m```.
- ***OCR Metrics:*** The index of 30 document (60 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for our OCR evaluation is ```$ROOT/ocr_img.txt``` (*Setting 1*). Please refer to [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) for the index of 25 document (50 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for their OCR evaluation (*Setting 2*). We provide the OCR evaluation code at ```$ROOT/OCR_eval.py```. The version of pytesseract is 0.3.8, and the version of [Tesseract](https://digi.bib.uni-mannheim.de/tesseract/) in Windows is recent 5.0.1.20220118. Note that in different operating systems, the calculated performance has slight differences.

 
|      Method             |    MS-SSIM   |      LD     |   Li-D   |  ED (*Setting 1*)  |       CER      |      ED (*Setting 2*)   |      CER       | 
|:-----------------------:|:------------:|:-----------:| :-------:|:----------------:|:--------------:|:---------------------:|:--------------:|
|    *DocScanner-T*       |     0.5123   |     7.92    |  2.04    |   501.82         |     0.1823     |    809.46             |     0.2068     | 
|    *DocScanner-B*       |     0.5134   |     7.62    |  1.88    |   434.11         |     0.1652     |    671.48             |     0.1789     | 
|    *DocScanner-L*       |     0.5178   |     7.45    |  1.86    |   390.43         |     0.1486     |    632.34             |     0.1648     | 
 
## Citation
If you find this code useful for your research, please use the following BibTeX entry.

```
@article{feng2021docscanner,
  title={DocScanner: Robust Document Image Rectification with Progressive Learning},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Tian, Qi and Li, Houqiang},
  journal={arXiv preprint arXiv:2110.14968},
  year={2021}
}
```

## Acknowledgement
The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) and [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet). Thanks for their wonderful works.
