# OCR on Indus Seals

## Abstract
Very serious and continuous attempts in deciphering the Indus Script of the Indus valley civilization has been carried out for years together, now, though many claim to have succeeded in deciphering it, there is still no standardized and universally accepted approach to it. All these years the humans have been employed in manually executing the process of decipherment, but, if the computers were able to read the indus script from the seals and other artifacts and recognize the symbols and the text sequences that they form, it will lead to a huge array of possibilities, that include, analysing the language, deciphering it and learning new additions to it, in an iterative way. For the computers to be able to read a new language, we present, in our work here, an Optical Character Recognition engine that is capable of reading the symbols of the Indus Script inscribed on the artifacts and be able to represent them in a computer readable format, say, a sequence of numbers in which each number corresponds to a symbol in the Indus Script according to the Mahadevan Corpus. With the advances in computer vision, image processing and deep learning, given images of Indus seals or other artifacts from the archaeological sites, our Optical Character Recognition engine tries to automatically locate text patches or regions from the images and further segment out the individual symbols from those regions. Once these symbols have been segmented out, we identify each symbol that belongs to the Indus Script and thus ultimately mining the symbol patterns that form the text. With these text sequences mined from the Indus Seal images and advanced machine learning techniques, we believe, to discover very interesting information and intriguing knowledge about the Indus Script and its semantics.

## Author Summary

## Introduction
The Indus valley civilization is the first major urban culture of South Asia, which was at its peak from 2600 BC to 1900 BC referred to as the Mature Harappan civilization. The Indus Script or the Harappan Script of the Indus valley civilization is referred to as a bronze age writing, which has still not been deciphered successfully, though many claim to have succeeded in it, all of their work claim, mutually exclusive statements and hence there is no universally accepted and standard decipherment till date. The reasons behind these failed attempts of decipherment are believed to be, because of the lack of bilingual text, inability to identify the underlying language and due the sparsely available script inscriptions. These Indus Scripts are extremely short inscriptions with an average length of 5 and maximum of 17 symbols per text inscription. They are usually inscribed on rectangular stamp seals and many other objects including tools, tablets, ornaments and pottery, made of many different materials such as soapstone, bone, shell, terracotta, sandstone, copper, silver and gold. According to Mahadevan corpus [] from the 3700 discovered seals, 417 signs have been found in specific patterns. Given these challenges to deciphering such an ancient language, which was for long, even being doubted if they constituted a writing system, that was later proved otherwise by showing that these languages have a syntactic structure like any other normal language such as English, based on statistical analysis[].

[literature survey TO DO]

## Materials and Methods

### Formulating the dataset
The dataset required for building this Indus Script OCR engine, includes the scans of almost all the Indus Seals discovered till date, with their corresponding text sequence decipherments according to the Mahadevan Corpus. However, unlike other computer vision problems, gathering the data for this particular use case is really challenging, as the data sources are very limited and small in size too. These sources include, the scans of the Indus Seals archived at the RMRL, Roja Muthiah Research Library and the seal images from scrapping the web via image search engines like Google Images. These were the only solid data sources used for building this OCR engine and the size of these datasets ranged in a few thousands, thus enforcing a constraint of building an efficient system that operates with minimal data. Using these primary data sources many different datasets where formulated to be used by the various stages of the OCR pipeline, which includes, The Indus Seals Dataset, The Text/NoText Dataset, The Symbols Dataset.

**The Indus Seals Dataset**, was the basic dataset formulated by combining the RMRL Indus Seal scans and web scraped Indus seal images. The RMRL Indus Seal scans, had the seals indexed from CISI M0101- M0620, which was nearly around 800 scans of the archived Indus Seals at the RMRL. Then, a web scrapper was built with Google Custom Search API to access Google Images and query for Indus Seal Images across the web. A maximum of 100 images per query term was retrieved to formulate this dataset, as going more than an offset of 100, only led to irrelavant images being pulled from the web. The query terms used for scrapping include: "indus seals", "harappan seals", "harappan pashupati seal", "harappan unicorn seal", "indus inscriptions", "harappan seals wikipedia", "indus seals and inscriptions", "indus seal stones", "seal impressions indus valley civilization", "indus valley tiger seals", "indus valley seals yoga", "indus valley seals for kids". Then from the crawled images we removed the noisy images manually and got 350 perfect seal images out of the 1000 images retrieved, this dataset is refered to as the, “The Indus Seals Dataset”, as a whole.

**The Text/NoText Dataset**, is used to build the “Text/NoText Classifier” of the Text region filtering module in the OCR pipeline, more information about this module discussed in below sections. This dataset was formulated by running the selective search algorithm for region proposal (discussed in the upcoming sections) over the images from “The Indus Seals Dataset” and then manually grouping the resulting regions into those containing(Text), not containing(NoText) and partly containing(Both) the Indus text. Thus the resulting in a dataset having three classes, namely, Text, NoText and Both and refered to as "The Text/NoText Dataset".

**The Symbols Dataset**, [TODO]

### Architectural Design
The Indus OCR engine, has various modules that form the sequential pipeline through which the input image is fed and the symbols are recognized. The architectural design has the following modules in sequence, Region Proposal module, Text Region Filtering module, Symbol Segmentation module and Symbol Identification module. These modules in-turn have sub modules as discussed below, that help achieve the intented task.

####Region Proposal
This module is responsible for extarcting the seals from the given image and coming up with possible regions of interest, that have a high possibilty of containing a symbol. This module has various sub modules as discussed below that help achieve this purpose. 

The **extract seal sub module**, takes the given image and removes the unnecessary background information, thus extracting the seal portion alone. In order to achieve the same, we smoothen the grey scaled image using Gaussian Blur, following this a thresholded connected components analysis is performed and the connected regions are labeled. We perform an optimized canny edge detection over this labeled image to get the edges, from which the contours are obtained and the bounding rectangular box around the seal is calculated.

The **selective search sub module**, is based on the object recognition algorithm, SelectiveSearch[]. The selective search is used as a region proposal algorithm, which proposes various regions of interests that is more likely to have an object within it, given an image. It is the fastest algorithm to compute ROIs and it combines the advantages of exhaustive search and segmentation, making it the optimal choice for out case. It basically performs hierarchical grouping of region proposals based on colour, texture, size, fill, etc. to propose the best ROIs. 

However, the raw ROI proposed by Selective Search does not succeed in satisfying our goal of extarcting text regions, so, in order to improve the region proposals by fine tuning the algorithm to suit our purpose, a greedy grid search approach over the 4 selective search parameters was performed to identify the best combination for a 512x512 resolution image. The various values tried for the four parameters are, Scale - 350, 450, 500 (higher the value larger the clusters in felzenszwalb segmentation[]), Sigma - 0.8 (Width of Gaussian kernel for felzenszwalb segmentation[]), Min Size - 30, 60, 120 (Minimum component size for felzenszwalb segmentation[]), Min Area - 2000(Minimum area of a region proposed). Once the parameters are fine tuned, the regions proposals were relevant enough but were really high in number and were mostly approximations and generalizations of each other.

Therefore,in order to reduce the number of regions proposed and to increase the quality of the region proposals, the following hierarchical grouping methods were devised and applied over the fine tuned selective search results, it is to be noted that, these were applied on images scaled to 512x512 or 256x256 or original size, that included four methodologies to group the basic region proposals. They are, **merge concentric proposals**, it was used to merge those proposals that were focusing on the same object with just small variations in the position and area being covered, and replaced them with the mean rectangle of all the concentric proposals. Then, **contained boxes removal** was performed to remove the proposals that were subsets of overall text regions or if some fraction of each symbol within a text region was also proposed along with the full text region, and only the overall proposals were retained. Further, the **draw super box** function was used to replace all the proposals that were overlapping each other such that a single symbol or text region was proposed as two different overlapping regions. The percentage overlap of such proposals was calculated and thresholded at 20 percent, all those pairs of regions having more than 20 percent overlap were replaced by a single minimal super box that bounded both the proposals. Finally, the **draw extended super box** function was used to replace those regions in hand now that were continuous subtext regions in the seal, arranged along the horizontal or vertical axes of the image. As all the subtext regions along the same axis belonged to a piece of text normally, all these were replaced by a single horizontal/vertical super box.

####Text region filtering (using Convolutional Neural Networks)

- ?Image augmentation and preprocessing?
- Text - no text classifier
- Filtering and Trimming region proposals
	
####Symbol segmentation

####Symbol Identification

- Jar sign experimentation
- Empirical Analysis of Pipeline’s Performance

- Evaluating the pipeline

## Results
## Discussion
## Conclusion
## Acknowledgments

## References
- wikipedia
- Mahadevan Corpus
- Statistical Analysis of the Indus Script Using n-Grams
- http://www.rmrl.in/
- http://koen.me/research/selectivesearch/
- https://www.cs.cornell.edu/~dph/papers/seg-ijcv.pdf
