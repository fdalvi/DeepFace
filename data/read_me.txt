original link: http://www.cs.columbia.edu/CAVE/databases/pubfig/download/

Overview

The PubFig dataset is divided into 2 parts:
The Development Set contains images of 60 individuals. This dataset should be used when developing your algorithm, so as to avoid overfitting on the evaluation set. There is NO overlap between this list and evaluation set, nor between this set and the people in the LFW dataset.
The Evaluation Set contains images of the remaining 140 individuals. This is the dataset on which you can evaluate your algorithm to see how it performs.
Due to copyright issues, we cannot distribute image files in any format to anyone. Instead, we have made available a list of image URLs where you can download the images yourself. We realize that this makes it impossible to exactly compare numbers, as image links will slowly disappear over time, but we have no other option. This seems to be the way other large web-based databases seem to be evolving. We hope to periodically update the dataset, removing broken links and adding new ones, allowing for close-to-exact comparisons.
Data Format

Almost all datafiles follow a "tab-separated values" format. The first two lines are generally like this:
# PubFig Dataset v1.2 - filename.txt - http://www.cs.columbia.edu/CAVE/databases/pubfig/
#    person    imagenum    url    rect    md5sum
The first line identifies the name and version of the dataset, the filename, and has a link back to this website. The second line defines the fields in the file, separated by tabs ('\t'). In this example (similar to the dev_urls.txt and eval_urls.txt files), there are 5 fields: person, imagenum, url, rect, and md5sum. The first two are common to many of the datafiles and are the name of the person and an image index number used to refer to a specific image of that individual. Note that image numbers are not necessarily sequential for each person -- there are "holes" in the counting.

Subsequent lines contain one entry per line, with field values also separated by tabs.

Development Set (60 people) - For algorithm development

Please use this dataset when developing your algorithm, to avoid overfitting on the evaluation set. You can create any type and number of training pairs from this dataset.

dev_people.txt: This contains a list of the 60 people in the development set. Each data line contains one person's name. There is NO overlap between this list and the people in the evaluation set nor between this set and the people in the LFW dataset.
dev_urls.txt: This contains URLs for all 16,336 images of the 60 people in the development set. (Because of copyright issues, we cannot distribute the images themselves.) Each data line is for one image and contains 5 elements, separated by tabs ('\t'):
the person name,
the image number for that person,
the original image url,
the face rectangle around the chosen person, as x0,y0,x1,y1 coordinates (x- and y-locations of the top-left and bottom-right corners of the face). Note that we only give the rectangle for the chosen person, even if there are multiple faces in the image.
the md5 checksum for the original image, as computed using the linux md5sum utility.

Evaluation Set (140 people) - ONLY for final performance evaluation

Please use this dataset ONLY when evaluating your algorithm, in preparation for submitting/publishing results. This is to prevent overfitting to the data and obtaining unrealistic results.

eval_people.txt: This contains a list of the 140 people in the evaluation set. The format is identical to that of the dev_people.txt file: Each data line contains one person's name.
eval_urls.txt: This contains URLs for all 42,461 images of the 140 people in the evaluation set. (Because of copyright issues, we cannot distribute the images themselves.) The format is identical to that of the dev_urls.txt file: Each data line is for one image, and contains 5 elements, separated by tabs ('\t'):
the person name,
the image number for that person,
the original image url,
the face rectangle around the chosen person, as x0,y0,x1,y1 coordinates (x- and y-locations of the top-left and bottom-right corners of the face). Note that we only give the rectangle for the chosen person, even if there are multiple faces in the image.
the md5 checksum for the original image, as computed using the linux md5sum utility.
pubfig_labels.txt: This contains some additional labels for each image in the evaluation set. Each data line contains the following fields, separated by tabs ('\t'):
the person name,
the image number for that person,
pose information as computed by our face detector, given as either frontal (both yaw and pitch within 10 degrees) or non-frontal
lighting information as labeled by users on Amazon Mechanical Turk, given as either frontal or non-frontal
expression information as labeled by users on Amazon Mechanical Turk, given as either neutral or non-neutral
pubfig_full.txt: Full verification benchmark of 20,000 images, divided into 10 cross-validation sets. Each set is mutually disjoint from all other sets, both by person and by image. During evaluation, you should use 9 of the sets for training, and the remaining 1 for testing. Then rotate through all 10 sets, so that in the end you have evaluated all pairs. Since each set is disjoint by identity, your evaluation algorithm will never have seen that person in training. Please do NOT use the filenames or person identities for anything other than reading the images! The format of this file is similar, but not identical, to that of the LFW benchmark:
The 1st line is a comment (starts with '#') identifying the file.
The 2nd line lists the number of cross-validation sets in this file (10 currently). After this follows each cross-validation set.
For each cross-validation set, the 1st line contains the number of positive and negative pairs within the set, separated by a tab.
This is then followed by the given number of positive examples (pairs of images of the same person), one per line. Each line contains 4 elements separated by tabs, for example:
Jodie Foster    81    Jodie Foster    220
These are:
The first person's name
The image number of the first person (as in eval_urls.txt)
The second person's name (for positive examples, this is the same as the first)
The image number of the second person (as in eval_urls.txt)
Finally, there are the given number of negative examples, in exactly the same format.
Miscellanous Files

These are some datafiles that cover both sets or might be helpful for related tasks.

pubfig_attributes.txt: A list of all 73 attribute values for all of the images in PubFig, computed using a newer version of our attribute classifiers. They are in the standard data format. Each data line contains the attribute values for a given image, referenced by person name and image index. Positive attribute values indicate the presence of the attribute, while negative ones indicate its absence or negation. The magnitude of the value indicates the degree to which the attribute is present/negated. The magnitudes are simply the distance of a sample to the support vector for the given classifier (using an RBF kernel). Thus, magnitudes should not be directly compared, even for the same attribute (and certainly not for different attributes). For details, please see the paper cited above.