//import opencv libraries 
#include <stdio.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\video\background_segm.hpp>

using namespace cv;
using namespace std;

void track(int, void*);
Mat img;
Mat mascara;
Mat img_gray, ventana, bordes, obtener;
int thresh = 140, maxVal = 255;
int type = 1, deger = 8;


int main() {

    //background subtractor is a technique that allows us to detect moving objects, 
    //while discarding the rest of the scene as it will remain motionless. 
    Ptr< BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
    cv::Rect myRoi(288, 12, 288, 288);
    Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    Mat frame;
    Mat resizeF;
    VideoCapture cap;

    //we open the camera of our pc to test the functionality 
    cap.open(0);
    while (1)
    {
        cap >> img;
        cv::flip(img, obtener, 1);
        cv::rectangle(obtener, myRoi, cv::Scalar(0, 0, 255));
        ventana = obtener(myRoi);
        cvtColor(ventana, img_gray, COLOR_RGB2GRAY);

        GaussianBlur(img_gray, img_gray, Size(23, 23), 0);
        //we define the variables of the trackbar windows 
        namedWindow("conjunto", WINDOW_AUTOSIZE);
        //call the track method created below 
        createTrackbar("Tipo Umbral", "conjunto", &type, 4, track);
        createTrackbar("Bordes", "conjunto", &deger, 100, track);
        createTrackbar("Limite", "conjunto", &thresh, 250, track);
        createTrackbar("Maximo", "conjunto", &maxVal, 255, track);

        pMOG2->apply(ventana, mascara);
        cv::rectangle(mascara, myRoi, cv::Scalar(0, 0, 255));

        track(0, 0);
        imshow("Imagen Original", obtener);
        imshow("Fondo Eliminado", mascara);
        imshow("Gris", img_gray);
        char key = waitKey(24);
        if (key == 27) break;
    }

    return 0;
}


//method to define masks and contours 
void track(int, void*) {
    int count = 0;
    char a[40];
    vector<vector<Point> > contorno;
    vector<Vec4i> hierarchy;

    GaussianBlur(mascara, mascara, Size(19, 19), 3.5, 3.5);
    threshold(mascara, mascara, thresh, maxVal, type);

    Canny(mascara, bordes, deger, deger * 2, 3);
    findContours(mascara, contorno, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    Mat dibujoo = Mat::zeros(bordes.size(), CV_8UC3);
    if (contorno.size() > 0) {
        size_t indexOfBiggestContour = -1;
        size_t sizeOfBiggestContour = 0;
        for (size_t i = 0; i < contorno.size(); i++) {
            if (contorno[i].size() > sizeOfBiggestContour) {
                sizeOfBiggestContour = contorno[i].size();
                indexOfBiggestContour = i;
            }
        }
        vector<vector<int> >hull(contorno.size());
        //the polygon that surrounds the hand according to the movement of the hand
        vector<vector<Point> >hullPoint(contorno.size()); 
        //green dots on fingertips ... multidimensional array
        vector<vector<Vec4i> > defects(contorno.size());
        //keeps the x, y points of the fingertip as point
        vector<vector<Point> >defectPoint(contorno.size());
        //keeps the x, y points of the fingertip as point
        vector<vector<Point> >contours_poly(contorno.size());

        Point2f rect_point[4];
        vector<RotatedRect>minRect(contorno.size());
        vector<Rect> boundRect(contorno.size());
        for (size_t i = 0; i < contorno.size(); i++) {
            if (contourArea(contorno[i]) > 5000) {
                convexHull(contorno[i], hull[i], true);
                convexityDefects(contorno[i], hull[i], defects[i]);
                if (indexOfBiggestContour == i) {
                    minRect[i] = minAreaRect(contorno[i]);
                    for (size_t k = 0; k < hull[i].size(); k++) {
                        int ind = hull[i][k];
                        hullPoint[i].push_back(contorno[i][ind]);
                    }
                    count = 0;

                    for (size_t k = 0; k < defects[i].size(); k++) {
                        if (defects[i][k][3] > 13 * 256) {
                            int p_start = defects[i][k][0];
                            int p_end = defects[i][k][1];
                            int p_far = defects[i][k][2];
                            defectPoint[i].push_back(contorno[i][p_far]);
                            circle(img_gray, contorno[i][p_end], 3, Scalar(0, 255, 0), 2);
                            count++;
                        }

                    }
                    //compares whether the sample is equal to the one entered 
                    if (count == 1)
                        strcpy_s(a, "1");
                    else if (count == 2)
                        strcpy_s(a, "2");
                    else if (count == 3)
                        strcpy_s(a, "3");
                    else if (count == 4)
                        strcpy_s(a, "4");
                    else if (count == 5 || count == 6)
                        strcpy_s(a, "5");
                    else
                        strcpy_s(a, "Muestra");

                    putText(obtener, a, Point(75, 450), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 255, 0), 3, 8, false);
                    //draw the windows we need to be visualized in the project 
                    drawContours(dibujoo, contorno, i, Scalar(255, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
                    drawContours(dibujoo, hullPoint, i, Scalar(255, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
                    drawContours(img_gray, hullPoint, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
                    approxPolyDP(contorno[i], contours_poly[i], 3, false);
                    boundRect[i] = boundingRect(contours_poly[i]);
                    rectangle(img_gray, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0);
                    minRect[i].points(rect_point);

                    for (size_t k = 0; k < 4; k++) {
                        line(img_gray, rect_point[k], rect_point[(k + 1) % 4], Scalar(0, 255, 0), 2, 8);
                    }

                }
            }

        }

    }
    imshow("Resultado", dibujoo);
}