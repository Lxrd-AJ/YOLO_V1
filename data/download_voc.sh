wget -c http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget -c http://pjreddie.com/media/files/VOC2012test.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOC2012test.tar

wget -c https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget -c https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

python voc_label.py