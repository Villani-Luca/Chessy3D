# Module objective
1) Find the corners of 0-N candidate chessboards (squared shapes)
2) For each candidate chessboard, detect the relative plan orientation with respect to the camera

# Processing steps
- Read the image
- Convert the image in grayscale
- Detect edges using Canny
- Detect lines using probabilistic Hough (HoughLinesP)
- Cluster lines into two groups, vertical and horizontal, using agglomerative hierarchical clustering
- Detect intersection between vertical and horizontal lines
- ....


# Log

## 2025-04-28
First attempt to detect edges using cv2.findChessboardCorners.
Failed because the algorithm tries to find the cells intersection points of empty chessboards, while the test images contain chess pieces.

## 2025-04-30
Tried the following pipeline to extract lines:
- cv2.cvtColor to convert the image to gray scale
- cv2.GaussianBlur to remove the noise
- cv2.Canny for the edges extraction
- cv2.HoughLinesP for the segments extraction

The Hough algorithm has two versions:
- HoughLines is used to detect lines in polar coordinates (infinite lines).
- HoughLinesP (probabilistic Hough) is used to detect segments delimited by two points in the Cartesian place.

cv2.HoughLinesP was chosen because it's more appropriate for real-world images.

The next step was to group the segments into vertical and horizontal.
A naive approach of using a treshold on segments angular coefficient doesn't work because the segments can be oriented in any possible direction inside the image.

The real first attempt was done performing clustering usign the angular coefficient of each segments as feature. The result was not correct because of two issues:

1) The angular coefficients of the vertical lines is inf (dy/dx, where dx=0)
2) The segment points provided by HoughLinesP are not oriented in the same direction, so lines which have a similar angular coefficient can be orientend backward (positive vs negative angular coefficient). This caused them to be grouped in different clusters.

The first problem was solved bu using the arctan to have a value mapped between [-π, π] instead of [-inf, +inf].

The second problem was solved by computing the absolute value of the features, grouping positive and negative numbers together, reducing the feature to a value in the range [0, π] where 0 and π are pure horizontal lines and π/2 are pure vertical line.

The clustering was performed using hierarchical agglomerative clustering, using the euclidian distance as distance metric and the ward linkage type, using a threshold which allows to have 2 clusters.

An alternative (not tested) options is to apply k-means with a cluster size of 2.