# NOVA IMS 2021/2022 - Deep Learning Project (Group 8)
The goal of this project is to build a Convolutional Neural Network that predicts the age and gender of a person by processing a picture of their face.

In essence, our model will receive a .jpeg image of the front of a person’s face and try to classify first as male, female or neutral (for babies that don't show very strong traces of any gender) and then in which bins of age this person is most probably is: 0-2, 3-6, 8-13, 15-24, 25-34, 35-43, 45-100. The bins of age are separated according to the available data that is taken from the OUI-Audience Face Image Project which compiled Flickr photos albuns uploaded from smartphones devices of people that allowed them to let their images to the general public.

For this project, we will mainly pursue higher accuracy rates as the main method of validation. Nevertheless, we also compare the loss, f1 score, overfitting and computing time between different models' architecture in order to provide a full view of how each version of the model improved.

Since this experiment already has a pre-processed database of images with the related labels, our group used the results of different papers with the similar task that used the same database in order to benchmark our work improvement.

References: 
Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
Eran Eidinger, Roee Enbar, and Tal Hassner, Age and Gender Estimation of Unfiltered Faces, Transactions on Information Forensics and Security (IEEE-TIFS), special issue on Facial Biometrics in the Wild, Volume 9, Issue 12, pages 2170 - 2179, Dec. 2014
T. Hassner, S. Harel, E. Paz, and R. Enbar. Effective face frontalization in unconstrained images. Proc. Conf. Comput. Vision Pattern Recognition, 2015.
