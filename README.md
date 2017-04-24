## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!


## Process:
This program was tough. Primarily because I did not know a lot of the techniques we had to use before doing it. I started
by trying to put all my code inside a Jupyter notebook, but that did not work out so well. I found that using a class was really nice,

### This project has three stages:
### Stage 1: Camera Distortion
* Our video stream was captured using a monocular camera. As such, we must correct the distortion that may occur. To do this, I used openCV's `drawChessboardCoenrs` and
`findChessboardCorners` functions. I then created two lists, `object_points` and `image_points`. Object points describe's where each pixel in the image exists in a real world representation (X, Y, Z).
Where image_points describe where each of those pixels are in the 2D .png image. We then compute the transformation matrix `mtx` and the distortion coefficient matrix `dist`, save them to a pickle file,
because later on in stage 3 we will recall and use them in `cv2.undistort()`.
* Code for this stage can be seen in Camera Calibration.ipynb

### Stage 2: Perspective Transformation:
* Our goal in this stage is to find source and destination points that we can use to warp our perspective to obtain an aerial view of the road.
* This will allow us to calculate the distance in meters per pixel between the lane lines
* It is also easier to work with the warped image inside the laneLine's objects, because we are dealing with straight vertical lines instead of angles lines.
* (it's easier to see if our predicted lines are well drawn compared to the given lanes).

* Steps in this stage
* Read in the image
* Undistort the image (using the transformation matrix and distortion coefficients from `Camera Calibration.ipynb`
* Convert the image to HLS
* Apply canny edge detection on the Lightness layer of the HLS image. (This gives us a better representation of the lane lines).
* Apply the Hough Transform to obtain coordinates for lines that are considered to be straight
#### Finding the vanishing point:
* The vanishing point in an image is the point where it looks like the picture can go on forever. If you recall photos of a sunset of a horizon where they appear to extend forever, that's the point we are looking for.
* a [vanishing](https://en.wikipedia.org/wiki/Vanishing_point) point is a point in the image plane where the projections of a set of parallel lines in space intersect.
* After we applied the Hough Transform we get coordinate sets inside a list
* The vanishing point is the the <strong>intersection</strong> of all the lines built from the coordinates inside that list
* It is the point with minimal squared distance from the lines in the hough list
* The total squared distance
# TODO: Insert eqn1 from /eqn/eqn1.png and label it eqn1
* where: I is the cost function, <strong>ni</strong> is the line normal to the hough lines and <strong>pi</strong> are the points on the hough lines
* Then we minimize I w.r.t vp.
# TODO: Insert eqn2 from /eqn/eqn2.png and label is eqn 2
### Important: How we found source points
To find the source points using the vanishing point `vp`, we had to be clever.
* So we have the vanishing point, which we can consider to be in the middle of our image. We want to locate a trapezoid that surrounds our lane lines.
* We first declare two points `p1` and `p2` which are evenly offset from the vanishing point, and are closer to our vehicle.
* Next, we want to find an additional two points `p3` and `p4` that exist on the line that connects p1 and vp, and p2 and vp.
* After that we will have the four points for our trapezoid (p1, p2, p3, p4), p1 and p4 live on the same line as do p2 and p3.
* We apply the equation (y2 - y1) = m (x2 - x1) and y = mx + b, and solve for x. (That is why we are using the inverse slope). This can be seen in `find_pt_inline`
* p1, p2, p3, and p4 form a trapezoid that will be our perspective transform region. our source points are [p1, p2, p3, p4].
* The source points define the section of the image we will use for our warping transformation
* The destination points are the where the source points will ultimately end up after we apply our warp. (pixels will be mapped from source points to destination points)

# TODO: Insert vanishing_point_visualization in /filtering/vanishing_point_visualization.png
# TODO: Here you can see the vanishing point as defined by the blue triangle

# TODO: Insert perspective transform visualization from /filtering/perspective_transform_visualization.png
* Here you can see the Trapezoid mask we will be using for our perspective transform. The source points are marked with the + and ^ dots.

#### Finding the distance
* We use moments in order to find the distance between two lane lines in our warped images.
* Lane lines are ~12 feet apart in the real world. We can use this info to find out how many pixels in our image equals 1 meter.
* We find the area between the lane lines using the zeroth moment, then we divide the first moment by the zeroth moment to get a centroid for both the right and left lane lines
* Now we find the minimum distance between the two centroids and define that to be our lane width. Then convert the lane width from feet to meters and now we have our pixels per meter in x.
* To find the pixels per meter in y, we normalize our homography matrix and extract the x and y components, then we multiply our x_pixels_per_meter by the projection of our x-norm in the y-direction. The projection provides the scale from x space to y space.
### x_pixel_per_meter:  53.511326971489076
### y_pixel_per_meter:  37.0398121547
* Then we save everything to a pickle file and move on to our lane line identification stage.
* Code for this stage can be seen inside Perspective_Transform.ipynb

### Stage 3: Lane Line Identification