🔥 ***2024.4.28:*** **Good news! The code and pre-trained model of DocScanner are now released!**

🚀 **Good news! The [online demo](https://docai.doctrp.top:20443/) for DocScanner is now live, allowing for easy image upload and correction.**

🔥 **Good news! Our new work [DocTr++: Deep Unrestricted Document Image Rectification](https://github.com/fh2019ustc/DocTr-Plus) comes out, capable of rectifying various distorted document images in the wild.**

🔥 **Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 

# DocScanner

<p>
    <a href='https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://docai.doctrp.top:20443/' target="_blank"><img src='https://img.shields.io/badge/Online-Demo-green'></a>
</p>


This is a PyTorch/GPU re-implementation of the paper [DocScanner: Robust Document Image Rectification with Progressive Learning](https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link).

![image](https://user-images.githubusercontent.com/50725551/209266364-aee68a88-090d-4f21-919a-092f19570d86.png)


## 🚀 Demo [(Link)](https://docai.doctrp.top:20443/)
***Note***：The model version used in the demo corresponds to ***"DocScanner-L"*** as described in the paper.
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.

<img width="1534" alt="image" src="https://github.com/fh2019ustc/DocScanner/assets/50725551/9eca3f7d-1570-4246-a3db-0a1cf1eece2d">

### Examples
![image](https://user-images.githubusercontent.com/50725551/223947040-eac8389c-bed8-433d-b23b-679c926fba8f.png)
![image](https://user-images.githubusercontent.com/50725551/223946953-3a46d6a3-4361-41ef-bb5c-f235392e1f88.png)


## Training
- We train the **Document Localization Module** using the [Doc3D](https://github.com/fh2019ustc/doc3D-dataset) dataset. Besides, [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset is exploited for background data enhancement.
- We train the **Progressive Rectification Module** using the [Doc3D](https://github.com/fh2019ustc/doc3D-dataset) dataset. Here we use the background-excluded document images for training.

## Inference 
1. Put the [pre-trained DocScanner-L](https://drive.google.com/drive/folders/1W1_DJU8dfEh6FqDYqFQ7ypR38Z8c5r4D?usp=sharing) to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Run the script and the rectified images are saved in `$ROOT/rectified/` by default.
    ```
    python inference.py
    ```

## Evaluation
- ***Important.*** In the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html), the '64_1.png' and '64_2.png' distorted images are rotated by 180 degrees, which do not match the GT documents. It is ingored by most of existing works. Before the evaluation, please make a check. Note that the performances in most of existing work are computed with these two ***mistaken*** samples.
- For reproducing the following quantitative performance on the ***corrected*** [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html), please use the geometric rectified images available from [Google Drive](https://drive.google.com/drive/folders/1QBe26xJwIl38sWqK2ZE9ke5nu0Mpr4dW?usp=sharing). For the ***corrected*** performance of [other methods](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification), please refer to the paper [DocScanner](https://arxiv.org/pdf/2110.14968v2.pdf).
- ***Image Metrics:***  We use the same evaluation code for MS-SSIM and LD as [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset based on Matlab 2019a. Please compare the scores according to your Matlab version. We provide our Matlab interface file at ```$ROOT/ssim_ld_eval.m```.
- ***OCR Metrics:*** The index of 30 document (60 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for our OCR evaluation is ```$ROOT/ocr_img.txt``` (*Setting 1*). Please refer to [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) for the index of 25 document (50 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for their OCR evaluation (*Setting 2*). We provide the OCR evaluation code at ```$ROOT/OCR_eval.py```. The version of pytesseract is 0.3.8, and the version of [Tesseract](https://digi.bib.uni-mannheim.de/tesseract/) in Windows is recent 5.0.1.20220118. Note that in different operating systems, the calculated performance has slight differences.

 
|      Method             |    MS-SSIM   |      LD     |   Li-D   |  ED (*Setting 1*)  |       CER      |      ED (*Setting 2*)   |  CER       |  Para. (M) |
|:-----------------------:|:------------:|:-----------:| :-------:|:----------------:|:--------------:|:---------------------:|:--------------:|:--------------:|
|    *DocScanner-T*       |     0.5123   |     7.92    |  2.04    |   501.82         |     0.1823     |    809.46             |     0.2068     |  2.6 |
|    *DocScanner-B*       |     0.5134   |     7.62    |  1.88    |   434.11         |     0.1652     |    671.48             |     0.1789     |  5.2 |
|    *DocScanner-L*       |     0.5178   |     7.45    |  1.86    |   390.43         |     0.1486     |    632.34             |     0.1648     |  8.5 |
 
## Citation
Please cite the related works in your publications if it helps your research:

```
@inproceedings{feng2021doctr,
  title={DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction},
  author={Feng, Hao and Wang, Yuechen and Zhou, Wengang and Deng, Jiajun and Li, Houqiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={273--281},
  year={2021}
}
```

```
@inproceedings{feng2022docgeonet,
  title={Geometric Representation Learning for Document Image Rectification},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Wang, Yuechen and Li, Houqiang},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

```
@article{feng2021docscanner,
  title={DocScanner: robust document image rectification with progressive learning},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Tian, Qi and Li, Houqiang},
  journal={arXiv preprint arXiv:2110.14968},
  year={2021}
}
```

## Acknowledgement
The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) and [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet). Thanks for their wonderful works.

## Contact
For commercial usage, please contact Professor Wengang Zhou ([zhwg@ustc.edu.cn](zhwg@ustc.edu.cn)) and Hao Feng ([haof@mail.ustc.edu.cn](haof@mail.ustc.edu.cn)).

