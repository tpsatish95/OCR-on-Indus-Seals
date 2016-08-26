# Deep learning the Indus Script

## Abstract
Very serious and continuous attempts in deciphering the Indus Script of the Indus valley civilization has been carried out for years together, now, though many claim to have succeeded in deciphering it, there is still no standardized and universally accepted approach to it. All these years the humans have been employed in manually executing the process of decipherment, but, if the computers were able to read the Indus script from the seals and other artifacts and recognize the symbols and the text sequences that they form, it will lead to a huge array of possibilities, that include, analyzing the language, deciphering it and learning new additions to it, in an iterative way. For the computers to be able to read a new language, we present, in our work here, an Optical Character Recognition engine that is capable of reading the symbols of the Indus Script inscribed on the artifacts and be able to represent them in a computer readable format, say, a sequence of numbers in which each number corresponds to a symbol in the Indus Script according to the Mahadevan Corpus. With the advances in computer vision, image processing and deep learning, given images of Indus seals or other artifacts from the archaeological sites, our Optical Character Recognition engine tries to automatically locate text patches or regions from the images and further segment out the individual symbols from those regions. Once these symbols have been segmented out, we identify each symbol that belongs to the Indus Script and thus ultimately mining the symbol patterns that form the text. With these text sequences mined from the Indus Seal images and advanced machine learning techniques, we believe, to discover very interesting information and intriguing knowledge about the Indus Script and its semantics.

## Author Summary

## Introduction
The Indus valley civilization is the first major urban culture of South Asia, which was at its peak from 2600 BC to 1900 BC referred to as the Mature Harappan civilization. The Indus Script or the Harappan Script of the Indus valley civilization is referred to as a bronze age writing, which has still not been deciphered successfully, though many claim to have succeeded in it, all of their work claim, mutually exclusive statements and hence there is no universally accepted and standard decipherment till date. The reasons behind these failed attempts of decipherment are believed to be, because of the lack of bilingual text, inability to identify the underlying language and due the sparsely available script inscriptions. These Indus Scripts are extremely short inscriptions with an average length of 5 and maximum of 17 symbols per text inscription. They are usually inscribed on rectangular stamp seals and many other objects including tools, tablets, ornaments and pottery, made of many different materials such as soapstone, bone, shell, terracotta, sandstone, copper, silver and gold. According to Mahadevan corpus [] from the 3700 discovered seals, 417 signs have been found in specific patterns. Given these challenges to deciphering such an ancient language, which was for long, even being doubted if they constituted a writing system, that was later proved otherwise by showing that these languages have a syntactic structure like any other normal language such as English, based on statistical analysis[].

A optical character recognition problem with similar complexities and challenges was done on Google's Street View Images[], which were natural photographs, that widely suffer from variations in fonts, orientations and lighting, shadows, low resolution and blur, to name a few. The paper has proposed a unified approach for localization, segmentation and recognition of multi-digit numbers from street view imagery using deep ConvNet architectures to train their models. The ConvNets output conditional probabilities of number sequences and the entire process is modeled as a sequence recognition task. As street view images, on an average, has numbers of not more than 5 digits in length, leading to 7 possible cases - 0,1,2,3,4,5 or "morethan 5 digits". If S is the output number sequence and X is the given image, they ultimately find p(S|X) as, P(S = s| X) = P(L = n| X) MultiplicationOverI[P(Si = si | X)]. In this, each of P(L = n|H) and P(Si = si|H) are derived from a SoftMax classifier trained on the input features, these SoftMax models can use backdrop learning and SGD to maximize log(P(S|X)) and at test time they find the ArgMax, for each length, each character, and ultimately each sequence, thus predicting the final sequence. This was all computationally possible as there only 10 possible digits and Google Street View had a huge dataset. Which is not true in our case. We have a vocabulary of roughly 417 symbols[] and a dataset of size very less than 4000 sample images. So this warrants the need for a newer architecture to be able to model this complex problem at hand, which is discussed in this paper.

## Materials and Methods

### Formulating the dataset
The dataset required for building this Indus Script OCR engine, includes the scans of almost all the Indus Seals discovered till date, with their corresponding text sequence decipherments according to the Mahadevan Corpus. However, unlike other computer vision problems, gathering the data for this particular use case is really challenging, as the data sources are very limited and small in size too. These sources include, the scans of the Indus Seals archived at the RMRL, Roja Muthiah Research Library and the seal images from scrapping the web via image search engines like Google Images. These were the only solid data sources used for building this OCR engine and the size of these datasets ranged in a few thousands, thus enforcing a constraint of building an efficient system that operates with minimal data. Using these primary data sources many different datasets where formulated to be used by the various stages of the OCR pipeline, which includes, The Indus Seals Dataset, The Text/NoText Dataset, The Symbols Dataset.

**The Indus Seals Dataset**, was the basic dataset formulated by combining the RMRL Indus Seal scans and web scraped Indus seal images. The RMRL Indus Seal scans, had the seals indexed from CISI M0101- M0620, which was nearly around 800 scans of the archived Indus Seals at the RMRL. Then, a web scrapper was built with Google Custom Search API[] in Python to access Google Images and query for Indus Seal Images across the web. A maximum of 100 images per query term was retrieved to formulate this dataset, as going more than an offset of 100, only led to irrelevant images being pulled from the web. The query terms used for scrapping include: "indus seals", "harappan seals", "harappan pashupati seal", "harappan unicorn seal", "indus inscriptions", "harappan seals wikipedia", "indus seals and inscriptions", "indus seal stones", "seal impressions indus valley civilization", "indus valley tiger seals", "indus valley seals yoga", "indus valley seals for kids". Then from the crawled images we removed the noisy images manually and got 350 perfect seal images out of the 1000 images retrieved, this dataset is referred to as the, “The Indus Seals Dataset”, as a whole.

**The Text/NoText Dataset**, is used to build the “Text/NoText Classifier” of the Text region filtering module in the OCR pipeline, more information about this module discussed in below sections. This dataset was formulated by running the selective search algorithm for region proposal (discussed in the upcoming sections) over the images from “The Indus Seals Dataset” and then manually grouping the resulting regions into those containing(Text), not containing(NoText) and partly containing(Both) the Indus text. Thus the resulting in a dataset having three classes, namely, Text, NoText and Both and referred to as "The Text/NoText Dataset".

**The Symbols Dataset**, [TODO]

### Architectural Design
The Indus OCR engine, has various modules that form the sequential pipeline through which the input image is fed and the symbols are recognized. The architectural design has the following modules in sequence, Region Proposal module, Text Region Filtering module, Symbol Segmentation module and Symbol Identification module. These modules in-turn have sub modules as discussed below, that help achieve the intended task.

####Region Proposal
This module is responsible for extracting the seals from the given image and coming up with possible regions of interest, that have a high possibility of containing a symbol. This module has various sub modules as discussed below that help achieve this purpose. 

The **extract seal sub module**, takes the given image and removes the unnecessary background information, thus extracting the seal portion alone. In order to achieve the same, we smoothen the gray scaled image using Gaussian Blur, following this a thresholded connected components analysis is performed and the connected regions are labeled. We perform an optimized canny edge detection over this labeled image to get the edges, from which the contours are obtained and the bounding rectangular box around the seal is calculated.

The **selective search sub module**, is based on the object recognition algorithm, SelectiveSearch[]. The selective search is used as a region proposal algorithm, which proposes various regions of interests that is more likely to have an object within it, given an image. It is the fastest algorithm to compute ROIs and it combines the advantages of exhaustive search and segmentation,like segmentation, the image structure is used to guide sampling and like exhaustive search, it captures all possible object locations invariant of size and scale, making it the optimal choice for our case. It basically performs hierarchical grouping of region proposals based on colour, texture, size, fill, etc. to propose the best ROIs. It was also used as the Region Proposal mechanism for R-CNN [by Girshick et al].

However, the raw ROI proposed by Selective Search does not succeed in satisfying our goal of extarcting text regions, so, in order to improve the region proposals by fine tuning the algorithm to suit our purpose, a greedy grid search approach over the 4 selective search parameters was performed to identify the best combination for a 512x512 resolution image. The various values tried for the four parameters are, Scale - 350, 450, 500 (higher the value larger the clusters in felzenszwalb segmentation[]), Sigma - 0.8 (Width of Gaussian kernel for felzenszwalb segmentation[]), Min Size - 30, 60, 120 (Minimum component size for felzenszwalb segmentation[]), Min Area - 2000(Minimum area of a region proposed). Once the parameters are fine tuned, the regions proposals were relevant enough but were really high in number and were mostly approximations and generalizations of each other.

Therefore,in order to reduce the number of regions proposed and to increase the quality of the region proposals, the following hierarchical grouping methods were devised and applied over the fine tuned selective search results, it is to be noted that, these were applied on images scaled to 512x512 or 256x256 or original size, that included four methodologies to group the basic region proposals. They are, **merge concentric proposals**, it was used to merge those proposals that were focusing on the same object with just small variations in the position and area being covered, and replaced them with the mean rectangle of all the concentric proposals. Then, **contained boxes removal** was performed to remove the proposals that were subsets of overall text regions or if some fraction of each symbol within a text region was also proposed along with the full text region, and only the overall proposals were retained. Further, the **draw super box** function was used to replace all the proposals that were overlapping each other such that a single symbol or text region was proposed as two different overlapping regions. The percentage overlap of such proposals was calculated and thresholded at 20 percent, all those pairs of regions having more than 20 percent overlap were replaced by a single minimal super box that bounded both the proposals. Finally, the **draw extended super box** function was used to replace those regions in hand now that were continuous subtext regions in the seal, arranged along the horizontal or vertical axes of the image. As all the subtext regions along the same axis belonged to a piece of text normally, all these were replaced by a single horizontal/vertical super box.

####Text region filtering (using Convolutional Neural Networks)

The curated region proposals from the previous stage of the pipeline have both text and non-text regions being proposed, sometimes a single region proposal might even have both text and non text portions in it, this makes handling such composite regions a tough task. Thus, these curated set of Regions of Interest in the given image containing a seal is fed into the region filtering module. The main idea of filtering these regions of interest, is to be able to, isolate out only those regions that have Indus text/symbols in them from all the other proposals that have non-text content or partial text/symbol content, in the latter case there involves some further trimming to get only the text out those composite region proposals, which is discussed in the later part of this section. 

Constructing this Text/No-Text filter, involves building a machine learned model that can learn the various characteristics that differentiate a text portion from a non-text portion, in any given Indus seal image. Now this makes up a Computer Vision challenge and in particular, it boils down to a Image classification problem, wherein we intend to classify the given images of regions of interest into three classes, namely, Text, No-Text and Both. This problem at hand might appear to be solvable by a very simple template matching or pattern matching algorithm, but, given the use case of Indus Scripts, this turns out to be really interesting problem. The Indus seals at our disposal and the ones to come out from archaeological sites are not as perfect in state as they where thousands of years ago, they have been subjected to serious wear and tear over time and have lost their originality, and what we have today are the remains of the great civilization, and performing image classification on such data, wherein most of the seals are broken, scratched, worn out, the seals and sealings add different elevations to the text, some seals are partially erased, they aren't uniform in scale, text size and spacing, the shot images of the Indus seals have different lighting conditions, symbols are vary similar to one another, we have nearly 417 symbols/alphabets [] of the language and finally we only have 3000+ images at our disposal to be able to learn a model robust to all such variations. 

The various image classification challenges that these complexities entail are, scale variation, deformation, illumination conditions, background clutter and intra class variations. Our goal was to build a image classification model that is invariant to the cross product of all these variations, while simultaneously retaining sensitivity to the inter-class variations. This is no usual image classification problem, where features used to model the images are hand crafted to suit the purpose, with complexer challenges at hand, instructing the computer where to look will not be viable solution, the possible solution to the problem is crafting deep learned features that are capable of adapting themselves on what to focus about and where to look in the images given the requirements and training samples, instead of hand-crafting it. It also enables learning a hierarchy of features with less human intervention, and all of this, with just minimal or no pre-processing.

The combination of deep learning and computer vision, calls for a very sought after machine learning algorithm, the Convolutional Neural Networks(CNN). The concept of CNNs was inspired from the actual working of the human eye and how the neurons processes images, with such biological relevance, it not only acts as a deep features extractor but as a classifier as well. Unlike other artificial neural networks the CNNs also have various layers, but not all the layers do the same task, there are different types of layers, the convolution layer, pooling layer, LRN[], ReLU[], dropout layer, full connected layer and SoftMax layer, to name a few. Each of these layers perform a specific task and have a number of hyper parameters that can be configured to suit the use-case at hand. The blueprint that describes how these layers are stacked together along with the hyper parameter configurations is referred to as the CNN's Architecture. 

Some of the famous CNN architectures used by image classification engines trained on huge datasets like the ImageNet[] are, LeNet[], AlexNet[], VGGNet[] and GoogLeNet[]. In general, the CNNs need datasets of size greater 1M data points, to learn and perform fairly well, but in our case the dataset size is very small, having only a few thousand data points. Moreover, there is no solid process to architect the CNN, it is a trial and error mechanism which is learned over time and thereby is considered an art.  This makes the idea of training a CNN from scratch with a new architecture, completely impossible. But, we can take the weights of an already trained CNN and fine tune it to suit our purpose of building a text filtering "text/no-text" classifier. 

On surveying the various CNN architectures[] for fine tuning purposes, we finalized on the, the most deepest, efficient and lightweight architecture, the GoogLeNet, which was designed for object classification and detection problems and best suits our needs. The GoogLeNet is a 22 layers deep network and has 9 Inception (network within a network) modules and was crafted as a submission for ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14) which achieved the state of the art results in the challenge[]. [IMG]

As discussed earlier, training a CNN from scratch is not a great idea with a comparatively small dataset, and given the size of CNNs, datasets like ImageNet take 3-4 weeks to train even with multiple GPUs. Therefore, pre-training a network on a larger dataset, and initializing those weights and filters before training with the primary dataset, is a very common practice and is referred to as Transfer Learning, as the weights and filters learned by the CNN on some other dataset is fine tuned and transfer learned to suit our dataset. In our case, we used the GoogLeNet's ImageNet trained final snapshot released by BVLC[] for unrestricted access for initializing our CNN, thus saving a lot of computational cycles and time. In our case, the Indus seals dataset at hand is small and is different from the ImageNet dataset, therefore using the freshly initialized CNN with ImageNet weights as a mere feature extractor is not appropriate and will lead to a poor model. So, we opt for fine-tuning the ConvNet.

Fine-tuning the ConvNet is based on the observation that the lower level layers of a CNN describe more generic features like Gabor Filters, blob detectors or edge detectors and are not specific to a particular dataset or task, but in general they are applicable to many datasets and tasks, but the later layers of the CNN becomes progressively more specific to the dataset. Therefore, this eventual transition of the features from general to specific by the last layer of the network, leads to initializing the GoogLeNet's CNN architecture with ImageNet trained weights from BVLC and retaining the rich lower level filters of the network by setting the learning rate for these layers to 0. Wherein, the later layers' learning rates where doubled than normal and the links to the final SoftMax layer was rewired to have just 3 output classes to suit our Indus text filtering use case.

These ConvNets where built, trained, transfer learned and fine-tuned using the Caffe deep learning framework developed by “Yangqing Jia” under Berkeley Vision. This is the most efficient and light weight implementations of the CNNs in C++, with support for multiple GPU based training, the other frame works like Tensor Flow, are less intuitive and 3 times slower than the Caffe implementation, though the Torch implementation is claimed to be equally fast and efficient, it cannot support multiple GPU training. The Caffe, uses the Google's protobuf to specify the CNN architectures as text configuration files, that are translated to in-memory CNN architectures by the frame work. The Caffe implementation with cuDNN trained on Nvidia GPUs is 40 times faster than basic CPU based training.

The text/no-text classifier built, labels the regions proposals as Text, No-Text or Both. Further, in order to generate more precise region proposals, trimming the region proposals off the partly non text regions is required. In order to achieve this, two methods where formulated, namely, Draw TextBox and Trim TextBox. The Draw TextBox method merges those pairs of region proposals, where a Text region and a Both region were overlapping, in order to get whole text regions, into a single TextBox. The Trim TextBox method clips off the non text regions in those pairs of region proposals, where a TextBox/Text region and a NoText were overlapping, in order to get the trimmed text regions clipped into a single TextBox. As final result of this filtering and trimming, we get the perfect text only region proposals.

- ?Image augmentation and preprocessing?
	
####Symbol segmentation

The precise text regions from the previous stage are fed as inputs into the Symbol Segmentation stage of the pipeline. The main aim of this module is to segment out the characters/symbols separately from the entire text region, for this purpose the Selective Search algorithm was not effective. So, a customized algorithm that stacked together various basic image processing techniques and thresholding techniques proposed by Otsu[], was devised. The various steps of the algorithm are described in [].

	- Segment_Symbol(Image I)
		- I = GrayScale(I)			 
		- I = OtsuThresholding(I)  	# Returns a discrete binary image
		- I = GaussianBlur(I) 		# Removes noise and smooth-ens the image 
		- Components = ConnectedColourComponents(I)	
									# Groups the connected subregions in the image, based on color
		- ROIs = Group(Components)	
									# Groups based on a combination of methods from the Region proposal and Text region filtering modules, previously
		- SubImages = Crop(ROIs, I)	# Crop the region proposals based on generated ROIs

This algorithm, extracts out the individual symbols and thus reducing the OCR problem to a symbol classification problem. 


####Symbol Identification

The individual character/symbol crops from the symbol segmentation module of the Indus OCR pipeline, are to be classified into one of the 417 classes of Indus symbols according to the Mahadevan Corpus (M77). In order to train such a ConvNet classifier from the meager data we have, we need to generate a more artificial data by augmenting the available base dataset with variations and noise using the, Image Preprocessing and Data Augmenter. This is an image pre-processing cum data augmenter, that we built in Python, inspired from Keras[] a deep learning library for Python with Google's TensorFlow backend. This is capable of performing various image pre-processing tasks like feature wise center, sample wise center, feature wise std normalization, sample wise std normalization and ZCA whitening. The data augmenter was developed to generate more data that could lead to a bigger and better dataset by performing the following operations on the base images we have, which include, vertical and horizontal flips, random shear, crop, swirl, rotate, scale and translate all done repeatedly with randomized parameters. This is believed to warrant towards building a more robust and generalized model for classifying Indus scripts.

As a first step to symbol identification, an experiment was conducted to detect the presence of the most frequently encountered Indus symbol, the Jar, from the given Indus seal images. This experimentation involves no other pre-processing, the Indus seals dataset collected by crawling the web was used for this experimentation. The data was manually classified into those having and not having the Jar sign in them. Then using a simple ConvNet OCR architecture[], as discussed in the upcoming results section, the model was successfully able to give a binary result about, as to whether the given image had a Jar sign or not with an accuracy of 93.76%.

## Results

- SGD solver’s parameters were modified such that the learning rate was 0.001 with a step size of 3200 and 16000 iterations.

- This gave a model with a recall of 93.76%, for text/no text classification of the ROIs

- Empirical Analysis of Pipeline’s Performance

- Evaluating the pipeline

- Jar sign exp
DATA[32*32*3] (scaled down by 256) -> Convolution[5x5 Kernel, Stride 1, 20 Outputs] -> Convolution[5x5 Kernel, Stride 1, 50 Outputs] -> Dropout (to prevent over-fitting) -> Fully Connected [500 Outputs] -> ReLU (Non linearity)-> Fully Connected [2 Outputs] (2 Classes) -> SoftMax Classifier.
For more info regarding the SGD Solver and Architecture (PFA the .prototxt files)
Accuracy: 92.07% (snapshot saved after 1000 iterations (58MB))

## Discussion
## Conclusion
## Acknowledgments

## References
- https://en.wikipedia.org/wiki/Indus_script
- The Indus Script: Texts, Concordance and Tables. Memoirs of the Archaeological Survey of India
- Statistical Analysis of the Indus Script Using n-Grams
- http://www.rmrl.in/
- http://koen.me/research/selectivesearch/
- https://www.cs.cornell.edu/~dph/papers/seg-ijcv.pdf
- http://cs231n.github.io/classification/
- www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
- http://cs231n.github.io/convolutional-networks/#architectures
- http://arxiv.org/abs/1411.1792
- http://cs231n.github.io/transfer-learning/
- https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
- http://bvlc.eecs.berkeley.edu/
- https://engineering.purdue.edu/kak/computervision/ECE661.08/OTSU_paper.pdf
- Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks - http://arxiv.org/abs/1312.6082
- Selective Search (Selective Search for Object Recognition, Uijlings et al.)[1]

[TODO]

- Survey for ROIs
-> Learning to propose objects (by Philipp et al.)[2]
-> Text Spotting (VGG, Oxford)[3]

- Problems faced and solved, refer mail for pictures

I was having a tough time updating our already trained model with the new images from RMRL, I was working with only set one (CISI M0101- M0200), which had 200+ images.

The problems I faced was, 
- The noise in the Text/NoText dataset I was building, I had to spend some time curating the ROI crops dataset, off the noise

- I came with a work around for the same, by using mean value subtraction technique over the images being used for building the model along with GooglNet's IMAGE_DATA layer of the CNN.

We have a 90% (89.3 %) accurate model as a result, which is very well trained, it works perfectly for the previous Google Images and also with the RMRL images, thus we have a more generalized model, though we seem to have sacrificed over the accuracy a bit from our old model.

Case 1:
The model's performance with the old Google Images, It works the same way it was before.
Inline image 4                                   Inline image 5
Inline image 6         Inline image 7

Case 2:
This is a perfect case, where the complete text region gets pulled out. As a result of generalization of our old model, last time itself.

Inline image 15      Inline image 16

Case 3:
This is the perfect case, but the CNN fails to recognize it as a text region, this is due to lack of data, previously, but now with all the new data available,we are able to pull them out successfully too. (Note: The portion on right was not classified as Text but perfectly last time, but now it is)

Inline image 9Inline image 10


Case 4:
This a case where only a partial text region get pulled out, it misses one or two symbols in the recognition process. (The crab like symbol to the right end is missed out in the cut), The partial text region itself was not got out as even a text part itself previously, now it is working too. Still we need to try getting the crab symbol alone, which is not a CNN problem, but a problem with selective search itself.

Inline image 13         Inline image 14

Case 5:
These are cases where the entire seal in the scan image that has padded white space around gets recognized as the text region. However, this can be solved if we apply this algorithm to the cut out region once again, which was already done in "auto_crop.py" !
(The whole seal was pulled out, but now we get this). But subsequently It only gets a partial text region out, it is a fail in the "Selective Search" which needs to be rectified, making it a Case 4 Problem now. But, I also took one more image("M-155 A") for this case, see below, it works perfect. Just this one seems to misbehave. We half the number of images under this category. 

The partly correct one.
Inline image 11                  Inline image 12

The perfect one, this too suffered from the whole image getting cropped at first,
Inline image 20                               Inline image 21

Case 6:
This was a very tough seal to crack last time, we didn't get any result.
This is the result we obtain this time! Which is really not bad at all !!
Inline image 17      Inline image 19                Inline image 18


So, from all this analysis, what I feel  now is, our Text/NoText classifier is nearly getting more mature. But some problems are in Case 4 due to the "Selective Search" misbehavior, which was persistent from our early stages itself, though it doesn't contribute to many cases. The case 6 is really tough to get right, but we are doing a decent job now with it.  And still I have only the Symbol segmentation part of the workflow alone to be tested. 