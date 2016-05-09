% title: OCR on Indus Seals
% subtitle: Satish Palaniappan, SSNCE
% author: Under the Guidance of,
% author: Dr. Ronojoy Adhikari, Institute of Mathematical Sciences, Chennai.
% thankyou: Thanks everyone!
% contact: <span>www</span> <a href="https://in.linkedin.com/in/satishpalaniappan">Satish Palaniappan</a>
% contact: <span>github</span> <a href="http://github.com/tpsatish95">tpsatish95</a>
% favicon: figures/favicon.ico
---
title: Overview
build_lists: true

**The Problem:**

- Statement
- Expected Output
- Architectural Design

---
title: Overview
build_lists: true

**Methodology:**

- Formulating the dataset
- Region proposal
    - Selective Search
    - Fine tuning and Region grouping
- Text region filtering (using Convolutional Neural Networks)
    - Text - no text classifier
    - Filtering and Trimming region proposals
- Symbol segmentation

---

title: Overview
build_lists: true

**The Results:**

- Evaluating the pipeline

**Future Prospects:**

- Symbol Identification
    - Jar sign experimentation
    - Empirical Analysis of Pipeline’s Performance
- Image augmentation and preprocessing

---

title: The Problem
class: segue dark nobackground

---

title: Problem Statement

<p style="text-align:justify">To automatically locate text patches/regions, segment individual symbols/characters from those regions and also identify each symbol/character belonging to the Indus Script, given images of Indus seals from archaeological sites, using image processing and machine learning techniques.</p>

---
title: Expected Output
class: img-top-center

<p style="text-align:justify">Given Indus seal images, we intend to extract the text sequences as shown below (mapping to the corresponding symbol numbers in the M77 Corpus),
</p>
<img src=figures/expected_output.png />

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design.png />

---
title: Overview of the system
class: img-top-center

<img src=figures/overview.png />

---
title: Methodology
class: segue dark nobackground

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design1.png />

---
title: Formulating the Dataset: Initial Steps

- A script written in Python was used to access the Google Custom Search API to get Indus Seal Images from the Google image search engine
- A maximum of 100 images per search term was retrieved
- **Search terms used:** <span style="font-size:70%">["indus seals", "harappan seals", "harappan pashupati seal", "harappan unicorn seal", "indus inscriptions", "harappan seals wikipedia", "indus seals and inscriptions", "indus seal stones", "seal impressions indus valley civilization", "indus valley tiger seals", "indus valley seals yoga", "indus valley seals for kids" ]
</span>

- Removed noisy images manually and got 350 useful images out of the 1000 images downloaded, this dataset is refered to as the, “crawled dataset”.

---
title: A snapshot of the crawled dataset
class: img-top-center

<img vspace="100px" height="300px" src=figures/crawled_data.png />

---
title: Text/No Text Dataset
class: img-top-center

<p style="text-align:justify; font-size: smaller"> This was formulated by running the selective search algorithm for region proposal (discussed in the upcoming slides) over the 350 seal images from the “crawled dataset” and then manually grouping the resulting 872 regions into those containing, not containing and partly containing the Indus text, to be used by the “Text/No Text Classifier” (discussed in upcoming slides).</p>

<img height="300px" src=figures/tnot_data.png />

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design2.png />

---

title: Region Proposal
subtitle: Selective Search Algorithm

- Given an image it proposes various regions of interests that is more likely to have an object within it
- It is the fastest algorithm to compute and it combines the advantages of exhaustive search and segmentation
- It performs hierarchical grouping of region proposals based on colour, texture, size, fill, etc. to propose the best ROIs

---

title: Selective Search - Working
class: img-top-center

<img height="400" src=figures/sel_search.png />

---
title: Selective Search - Fine Tuning

In order to improve the region proposals to suit the purpose of identifying text regions in seal images, a greedy grid search approach over 4 parameters was performed to identify the best combination for a 512x512 image

- **Scale** - Higher the value larger the clusters in felzenszwalb segmentation (350, 450, 500)
- **Sigma** - Width of Gaussian kernel for felzenszwalb segmentation (0.8)
- **Min Size** - Minimum component size for felzenszwalb segmentation (30, 60, 120)
- **Min Area** - Minimum area of a region proposed (2000)

---

Once these parameters were fine tuned the regions proposed were relevant enough but were really high in number. Also they were mostly approximations and generalizations of each other.

<img height="400" src=figures/sel_search1.png />

---

title: Selective Search - Customised Grouping

<p style="text-align:justify; font-size: smaller"> In order to reduce the number of regions proposed and to increase the quality of the region proposals the following hierarchical grouping methods were devised, (These were applied on images scaled to 512x512 or 256x256 or original size): </p>

- <p style="text-align:justify; font-size: smaller"> **Merge Concentric Proposals:** Most of the proposals were focusing on the same object with just small variations in the position and area being covered. Such proposals were merged together and replaced by the mean rectangle of all the concentric proposals </p>

- <p style="text-align:justify; font-size: smaller"> **Contained Boxes Removal:** Some other proposals were subsets of overall text regions, some fraction of each symbols within a text region was also proposed along with the full text region. So, all of these subsets were removed and only the overall proposals were retained </p>

---


- <p style="text-align:justify; font-size: smaller"> **Draw Super Box:** Other proposals were overlapping each other such that a single symbol or text region was proposed as two different overlapping regions. The percentage overlap of such proposals was calculated and thresholded at 20 percent, all those pairs of regions having more than 20 percent overlap were replaced by a single minimal super box that bounded both the proposals
</p>

- <p style="text-align:justify; font-size: smaller"> **Draw Extended Super Box:** The regions in hand now were continuous subtext regions in the seal, arranged along the horizontal or vertical axes of the image. As all the subtext regions along the same axis belonged to a piece of text normally, all these were replaced by a single horizontal/vertical super box
</p>

---

title: Region Proposal - Output
class: img-top-center

<img height="330px" src=figures/sel_search2.png />

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design3.png />

---

title: Text Region Filtering using CNN
subtitle: Why Text region filtering?

From the results of our region proposal mechanism discussed above, it is clear that the final region proposals have both text and nontext regions being proposed, sometimes a single region proposal might even have both text and non text portions in it. In order to filter these proposals, a CNN image classifier is going to be trained, which will facilitate the region filtering and trimming

---
title: Convolutional Neural Network

- The concept of CNN is inspired from the actual working of the human eye and how neurons processes images
- The main strength of using a CNN is that it works as a feature extractor with deep learnt features as well as a classifier
- The Caffe deep learning framework developed by “Yangqing Jia” under [Berkeley Vision](http://bvlc.eecs.berkeley.edu/), was used to build the CNNs

---
title: A sample CNN Architecture
class: img-top-center

<img src=figures/cnn1.png />

---
title: Why Deep Features?

<p style="text-align:justify"> Generally, the way images are vectorized into features has always been hand crafted, but now with increasing problem complexity these deep learnt features are capable of adapting themselves on focusing about, what to look in the images given the requirements, instead of hand-crafting it. With just minimal or no pre-processing, a hierarchy of features can be learnt and with less effort.</p>

---
title: Features samples learnt at various stages of CNN
class: img-top-center

<img src=figures/cnn2.png />

---
title: CNN Architecture

- The CNNs consist of a variety of layers like
    - Convolution, Pooling, LRN, ReLU, Dropout, Full Connected, SoftMax Classifier
    - each of these layers have a number of parameters that can be configured.
- This blueprint that describes the hierarchy of the layers and the parameter configurations is referred to as the CNN Architecture.

---
title: Classifiers trained using CNN (1)

- **Jar Symbol Binary Classifier**
    - **Dataset** -  The Jar Sign Dataset
    - **Architecture** -  The pannous/caffe-OCR CNN OCR Architecture
    -  **Purpose** - To detect the presence of the most frequently encountered Indus symbol, the Jar, from the given Indus seal images
    - **Note:** This model has no application in the OCR pipeline being discussed currently and was just an experiment

---
title: Classifiers trained using CNN (2)

- **Text/No Text Classifier**
    - **Dataset** - The Text/No Text Dataset
    - **Architecture** - The GoogLeNet CNN Architecture
    - **Purpose** - To classify regions proposed by the Selective Search algorithm in the previous step, into 3 main classes namely - Text, No Text and Both, thus facilitating region filtering and trimming

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design4.png />

---
title: Text/No Text Classifier

- From the results of our region proposal, it is clear that the final region proposals have both text and nontext regions being proposed, sometimes a single region proposal might even have text and non text portions in it. In order to filter these proposals, a CNN image classifier was built by fine tuning the [GoogLeNet CNN Architecture ](http://arxiv.org/abs/1409.4842) trained on ImageNet images to suit our needs. There were three classes considered, Text, No Text and Both. The “Text/No text Dataset” was used for this purpose.
- **Dataset size:** Text - 232, No-Text - 543, Both - 97 images respectively, 70:30 stratified split for train and test was used

---
title: Text/No Text Classifier - GoogLeNet Architecture

- GoogLeNet is a 22 layers deep network
- It is one of the most efficient and powerful implementations of CNN
- It was mainly designed for object classification and detection problems
- It has 9 Inception modules (Network within a network)

---
title: GoogLeNet - Inception Module
class: img-top-center

<img height="400px" src=figures/cnn_inception.png />

---
title: GoogLeNet - Architecture
class: img-top-center

<img src=figures/cnn_arch.png />

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design5.png />

---
title: Using pre-learnt GoogLeNet weights

- GoogLeNet was designed for extremely large datasets and hence fails to deliver high accuracy for datasets as small as 872 images
- A concept called, <i>Transfer Learning</i> comes into picture, where the weights and filters learnt by the convolutional neural networks on some other dataset could be fine tuned and transfer learnt to suit our dataset
- In order to achieve transfer learning in our case, the GoogLeNet was initialized with the pre-trained weights and filters that were trained on the [ImageNet](http://www.image-net.org/) images

---
title:Fine Tuning the Network

- The initial layers of the network had the rich low level filters of the ImageNet images initialized.
- The learning rate for these layers was set to 0 and as the network progressed into its depth the already available base learning rates were doubled.
- Links to the SoftMax classifier was initialized to random weights with 3 outputs in our case.
- SGD solver’s parameters were modified such that the learning rate was 0.001 with a step size of 3200 and 16000 iterations.

---
title: GoogLeNet - Solver Configuration
class: img-top-center

<img height="400px" src=figures/cnn3.png />

---
title: Text Filtering  - Results Obtained
class: img-top-center

This gave a model with a recall of 93.76%, for text/no text classification of the ROIs
<img height="300px" src=figures/table1.png />

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design6.png />

---
title: Filtering and trimming region proposals

After getting the region proposals from selective search and their labels (Text, No Text and Both) from the Text/No Text Classifier, the following two methods were applied to generate much more accurate and crisp text regions.

- **Draw TextBox:** In some pairs of region proposals, where a Text region and a Both region, were overlapping, In order to get the whole text regions, the overlapping boxes were merged to a single text box
- **Trim TextBox:** In some pairs of region proposals, where a Text Box region and a NoText, were overlapping, In order to get the trimmed text regions, the overlapping boxes were clipped to a single text box

---
title: Trimming Region Proposals - Results Obtained
class: img-top-center

<img src=figures/trim.png />

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design7.png />

---
title:Symbol Segmentation (1)

Once we have the text regions, we need to segment out the characters/symbols, for that purpose the selective search algorithm was not effective, So, a customized algorithm that involved the the following steps was used to get the individual symbols out of the text regions,

- Gray scale the image
- Black and White Thresholding [by Otsu] to get a discrete binary image (easy to segment)
- Gaussian Blur to remove noise

<p style="text-align: right">continued...</p>

---
title:Symbol Segmentation (2)

- SciPy's find_object() method to get the connected subregions in the image
- Region grouping mechanisms discussed previously applied with slight modifications
- Incorporate these regions over the original image

---
title:Symbol Segmentation - Results Obtained
class: img-top-center

<img src=figures/sym1.png />


---
title: The Results
class: segue dark nobackground

---
title: Evaluating the Pipeline (1)

- In order to evaluate the performance of the text localization and segmentation pipeline discussed above, 25 unique seal images were taken from Google and was passed through the Indus OCR pipeline.
- **Results:**
    - **Text Localization:** Text regions from 23 out of 25 images were extracted successfully
    - However, In 5 of those 23 images the proposed text regions missed one symbol 92% Accuracy (Approximate)

<p style="text-align: right">continued...</p>

---
title: Evaluating the Pipeline (2)

- **Symbol extraction:** Out of those 23 images, symbols were successfully extracted from 14 images.
    - Out of the 9 failed images, 5 images were of low quality and were blurred
    - The remaining 4 images of those 9, failed due to physical damages in the seals

---
title: Results - Perfectly Working
class: img-top-center

<img height="230px" src=figures/result1.png />

---
title: Results - Failed Cases
class: img-top-center

<img height="230px" src=figures/result2.png />

---
title:Empirical Analysis of Pipeline’s Performance (1)

The symbol extraction pipeline has some drawbacks in terms of performance, it fails to perfectly extract symbol regions from the indus seal images, due to performance dropout at various stages of the pipeline, let us see where and why are there performance degrades,

- **Text Localization**
    - **Problem:**  The text region proposed, misses out one symbol or fails to localize the text patch
    - **Reason:** Blurred images, Complex seal structure, Contrasting / Bad lighted pictures

<p style="text-align: right">continued...</p>

---
title:Empirical Analysis of Pipeline’s Performance (2)

- **Symbol Extraction**
    - Problem: The symbol extraction module failed to extract complete symbol regions
    - Reason: Physical damages in seals , Low quality and blurred images

---
title: Future Prospects
class: segue dark nobackground

---
title: Architectural design of the proposed system
class: img-top-center

<img src=figures/arch_design8.png />

---
title: Symbol Identification

- After perfecting the symbol segmentation module of the above discussed Indus OCR pipeline, we need to identify each symbol and classify them into one of the 417 classes of Indus symbols according to the Mahadevan Corpus(M77)
- For training such a CNN classifier, we need to generate a more robust dataset augmenting the available base dataset with noise and gather more seal images labeled with corresponding text sequences
- As a prior experimentation, we wanted to build a classifier that was capable of identifying, the presence of the most frequently spotted Indus symbol, the JAR sign, in the Indus Seal images, called the Jar sign experimentation.

---
title: Jar sign experimentation - Dataset Formulation
class: img-top-center

<p style="text-align: justify"> <b>The Jar Sign Dataset:</b> This was formulated by manually classifying the 350 seal images from the “crawled dataset”, into those containing and not containing the Jar sign, to be used by the “Jar sign binary classifier”. </p>

<img src=figures/jarsign.png />

---
title: Jar sign experimentation - Training the Classifier

- The Jar Symbol Binary Classifier was built to detect the presence of Jar sign in the Indus seal image
- For this purpose a CNN architecture named IndusNet, inspired from [“pannous/caffe-OCR”](https://github.com/pannous/caffe-ocr) was built. The skeleton is shown below,
    - <p style="text-align: justify; font-size: 65%">DATA\[32x32x3\](scaled down by 256) -> Convolution[5x5 Kernel, Stride 1, 20 Outputs] -> Convolution[5x5 Kernel, Stride 1, 50 Outputs] -> Dropout (to prevent over-fitting) -> Fully Connected [500 Outputs] -> ReLU (Non linearity)-> Fully Connected \[2 Outputs\](2 Classes) -> SoftMax Classifier </p>
- The dataset was formulated by applying a 70:30 stratified split over the “Jar sign dataset”
- The model built had an accuracy of 93.4% after 1000 iterations

---
title:Data Augmentation and Pre-processing

- **Data Augmentation:** An image data augmentation tool was developed to achieve the same. The basic code was inspired from the online image data augmenter of [“Keras“](http://keras.io/), which included,
    - <p style="text-align: justify">Horizontal and vertical flip, Rotation, scaling, translation, Shear, Swirl, Blur, Contrast, Brightness (TO DO)</p>
- **Pre-processing:**  Some of the other common image pre-processing methods to be used are feature-wise mean-zero and standard normalization, sample-wise mean-zero and standard normalization
