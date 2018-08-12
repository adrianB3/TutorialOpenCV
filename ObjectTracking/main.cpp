#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>
#include <thread>
#include <chrono>
using namespace cv;
using namespace std;

#define SSTR(x) static_cast<std::ostringstream &>((ostringstream() << dec << x)).str()

int main(int argc, char **argv) {

    // Tracker types provided by OpenCV

    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW","GOTURN", "MOSSE", "CSRT"};

    // create tracker

    string trackerType = trackerTypes[2];

    Ptr<Tracker> tracker;

#if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
#else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
        if (trackerType == "MOSSE")
            tracker = TrackerMOSSE::create();
        if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();
    }
#endif
    VideoCapture capture;
    Mat frame;

    capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
    while(waitKey(1) != 27){
        bool ok = capture.read(frame);
        imshow("Tracking", frame);
    }
    // Define initial bounding box
    Rect2d bbox(287, 23, 86, 320);

    bbox = selectROI(frame, false);
    // Display bounding box.

    imshow("Tracking", frame);
    tracker->init(frame, bbox);

    while(capture.read(frame)){
        double timer = (double)getTickCount();

        bool ok = tracker->update(frame, bbox);

        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        if(ok){
            // Tracking success
            rectangle(frame, bbox, Scalar(255,0,0), 2, 1);
        }
        else {
            putText(frame, "Tracking failure", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }
        putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        putText(frame, "FPS: " + SSTR(int(fps)) , Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

        imshow("Tracking", frame);
        if(waitKey(1) == 27){
            break;
        }
    }

    return 0;
}