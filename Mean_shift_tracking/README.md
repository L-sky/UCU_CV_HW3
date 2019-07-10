# Datasets 

## Tracking
Visual Tracker Benchmark: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

Used: Board, BlurFace, Skating2, BlurCar2, Girl.

## Clustering
P. Fränti and S. Sieranoja "K-means properties on six clustering benchmark datasets": http://cs.joensuu.fi/sipu/datasets/

Used: S-sets, G2-sets. 


# Results

## Tracking
Own implementation of MeanShift and CamShift behaves very close yet not identically to OpenCV implementation. That most likely is a result of different covering of edge cases, such as apparent loss of object. E.g.: in my implementation, when MeanShift loss object (total probability in ROI is too low) it considers the whole image.  

In all cases, refered both own and OpenCV implementations of MeanShift and Camshift respectively, unless stated otherwise.
Images converted to hsv, and only hue channel used. In case of "template" to calculate histogram, and for other dataset images to make back projection given this histogram.   

**Board**: all algorithms start well, until circuitboard reaches right side of the screen, then they stuck in right lower corner, which means that for some reason probability of locating object in this region is not particularly low. Also, background mostly static, so once trapped, not much chances to get out, however MeanShift do, while CamShift - not. I suspect this is because CamShift is more biased towards its current location.

**BlurFace**: CamShift work like charm. As for MeanShift, it looks shifted visually, I think that is due to different scale of the face on the very first frame (smaller) and all others.

**Skating2**: all algorithms fail miserably. It seems they are more keen to stick to certain parts of background, which are barely present on "template". I think the reason here is that person rotating here a lot, and we almost immediately lose "known side" from eyesight of camera, and there is not enough correspondance between front-view, back-view and side-view for algorithms to keep the track. 

## Clustering 
MeanShift successfully performs clustering (granularity tuned with window parameter), for detailed results refer to the mean_shift_clustering.ipynb
