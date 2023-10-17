# Link prediction project
Repository for scientific project in Petnica Science Center (2019)

![](graph_414.png)

## Abstract

This paper explores what data is best to train a classifier that predicts friendships in social networks. The accuracies of the classifiers that were trained by user data (node attributes) and similarity metrics (network topology)were compared. Similarity metrics were better for smaller network, because classifiers trained by similarity metrics gave an average accuracy of 75.65% and classifiers trained by node attributes accuracy of 60.87%. For larger social network, the situation is reversed, as classifiers trained by similarity metrics give an average accuracy of 81.46%, and classifiers trained by node attributes accuracy of 87.17%. From this, we can conclude that it is better to use information about users than similarity metrics to train classifiers for large social networks. Node attributes and similarity metrics were also analyzed to see which attributes and which metrics are the best in prediction. Attributes that are related to user education and job are the best for predicting friendships. As for the similarity metrics, *Resource Allocation* metric is the most important metric in predictions that use similarity metrics.

## Paper

Full paper is written in Serbian and was presented at Petnica conference "A step into the science" in November 2019.

It was later published in: _Petnica Papers 2019, p. 315-322 ISBN 978-86-7861-057-8 (ISSN 0354-1428)_

[Link to paper](https://esveske.github.io/pdf/2019/PS78-Zb2019_I-RACh-NK.pdf)

## References

[1] Adamic L. A., Adar E. 2003. Friends and neighbors
on the Web. Social Networks, 25 (3): 211.

[2] Al Hasan M., Chaoji V., Salem S., Zaki M. 2006.
Link prediction using supervised learning. U
Proceedings of SDM 06: workshop on link analysis,
counter-terrorism and security. SIAM, str. 798-805.

[3] Gao F., Musical K., Cooper C., Tsoka S. 2015. Link
prediction methods and their accuracy for different
social networks and network metrics. Scientific
Programming, 2015: 1-13.

[4] Jaccard P. 1912. The distribution of the flora in the
alpine zone. New phytologist, 11 (2): 39.

[5] Julian K., Lu W. 2016. Application of machine
learning to link prediction. http://cs229.stanford.edu

[6] Leskovec J., Krevl A. 2014. SNAP Datasets:
Stanford Large Network Dataset Collection.
http://snap.stanford.edu/data

[7] Liben-Nowell D., Kleinberg J. 2004. The link
prediction problem for social networks. https://www.
cs.cornell.edu/home/kleinber/link-pred.pdf