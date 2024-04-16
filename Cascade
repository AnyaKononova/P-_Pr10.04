#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <mutex>

using namespace cv;
using namespace std;
using namespace std::chrono;

CascadeClassifier faceCascade, eyesCascade, smileCascade;
mutex videoMutex;

void detectAndDraw(Mat& frame, Mat& gray) {
    vector<Rect> faces;
    faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const auto& face : faces) {
        rectangle(frame, face, Scalar(255, 0, 0), 2);
        Mat faceROI = gray(face);

        vector<Rect> eyes;
        eyesCascade.detectMultiScale(faceROI, eyes);

        for (const auto& eye : eyes) {
            Point eye_center(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
            int radius = cvRound((eye.width + eye.height) * 0.25);
            circle(frame, eye_center, radius, Scalar(0, 255, 0), 2);
        }

        vector<Rect> smiles;
        smileCascade.detectMultiScale(faceROI, smiles, 1.165, 35, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (const auto& smile : smiles) {
            rectangle(frame, Point(face.x + smile.x, face.y + smile.y),
                Point(face.x + smile.x + smile.width, face.y + smile.y + smile.height),
                Scalar(0, 0, 255), 2);
        }
    }
}


void processVideo(VideoCapture& cap, VideoWriter& videoWriter) {
    while (true) {
        Mat frame;
        {
            lock_guard<mutex> lock(videoMutex);
            cap >> frame;
        }
        if (frame.empty()) {
            cout << "End" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        detectAndDraw(frame, gray);

        {
            lock_guard<mutex> lock(videoMutex);
            videoWriter.write(frame);
        }

    }
}

int main() {
    if (!faceCascade.load("haarcascade_frontalface_alt.xml") ||
        !eyesCascade.load("haarcascade_eye_tree_eyeglasses.xml") ||
        !smileCascade.load("haarcascade_smile.xml")) {
        cout << "Error loading Haar cascades!" << endl;
        return -1;
    }

    string videoPath = "C:/Users/Nuta/Documents/Open CV/09.04/task1/task1/video.mp4";

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cout << "Error opening video file!" << endl;
        return -1;
    }

    string outputVideoPath = "C:/Users/Nuta/Documents/Open CV/09.04/task1/task1/output_video.avi";
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter videoWriter(outputVideoPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

    auto start = high_resolution_clock::now(); 

#pragma omp parallel num_threads(4)
    {
        processVideo(cap, videoWriter);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    cout << "Time taken: " << duration.count() << " sec" << endl;

    return 0;
}
