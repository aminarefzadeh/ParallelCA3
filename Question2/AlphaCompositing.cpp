#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "x86intrin.h"
#include <sys/time.h>
#include <string>

using namespace std;

#define SID_1 810195424
#define SID_2 810195578

cv::Mat zeroPadding(cv::Mat &inputImage) {
  int padding = (16 - (inputImage.cols % 16)) % 16;

  cv::Mat paddedImage(inputImage.rows, inputImage.cols + padding, inputImage.type());
  paddedImage.setTo(cv::Scalar::all(0));

  inputImage.copyTo(paddedImage(cv::Rect(0, 0, inputImage.cols, inputImage.rows)));
  return paddedImage;
}

cv::Mat cropZeroes(cv::Mat &inputImage, int padding) {
  cv::Rect cropRegion(0, 0, inputImage.cols - padding, inputImage.rows);
  return inputImage(cropRegion);
}

int main() {
  printf("Amin Arefzade %d\nAmirhossein Mahmoodi %d\n", SID_1, SID_2);

  struct timeval start, end;
  long serialTime, parallelTime;

  cv::Mat background = cv::imread("CA03__Q2__Image__01.png", cv::IMREAD_GRAYSCALE);
  cv::Mat foreground = cv::imread("CA03__Q2__Image__02.png", cv::IMREAD_GRAYSCALE);

  int colsBeforePadding = background.cols;
  background = zeroPadding(background);
  foreground = zeroPadding(foreground);
  int padding = background.cols - colsBeforePadding;

  unsigned int backgroundRows = background.rows;
  unsigned int backgroundCols = background.cols;
  unsigned int foregroundRows = foreground.rows;
  unsigned int foregroundCols = foreground.cols;

  cv::Mat serialOutput (backgroundRows, backgroundCols, CV_8U);
  cv::Mat parallelOutput (backgroundRows, backgroundCols, CV_8U);

  unsigned char *serialBackgroundFD = (unsigned char *) background.data;
  unsigned char *serialForeground = (unsigned char *) foreground.data;
  unsigned char *serialFD = (unsigned char *)serialOutput.data;

  gettimeofday(&start, NULL);
  for (int row = 0; row < backgroundRows; row++) {
    for (int col = 0; col < backgroundCols; col++) {
      if (row < foregroundRows && col < foregroundCols)
        {
          *(serialFD + row * backgroundCols + col) = *(serialBackgroundFD + row * backgroundCols + col) +
                                                    (*(serialForeground + row * foregroundCols + col) >> 1);
          if(*(serialFD + row * backgroundCols + col) < *(serialBackgroundFD + row * backgroundCols + col))
            *(serialFD + row * backgroundCols + col) = -1;
        }
      else
        *(serialFD + row * backgroundCols + col) = *(serialBackgroundFD + row * backgroundCols + col);
    }
  }
  gettimeofday(&end, NULL);
  serialTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

  __m128i *parallelBackgroundFD = (__m128i *)background.data;
  __m128i *parallelForegroundFD = (__m128i *)foreground.data;
  __m128i *parallelFD = (__m128i *)parallelOutput.data;
  __m128i xReg1, xReg2;
  __m128i mask = _mm_set1_epi8(0x7f);

  gettimeofday(&start, NULL);
  for (int row = 0; row < backgroundRows; row++) {
    for (int col = 0; col < backgroundCols / 16; col++) {
      xReg1 = _mm_loadu_si128(parallelBackgroundFD + row * backgroundCols / 16 + col);
      if(row < foregroundRows && col < foregroundCols / 16) {
        xReg2 = _mm_loadu_si128(parallelForegroundFD + row * foregroundCols / 16 + col);
        xReg2 = _mm_srli_epi16(xReg2, 1);
        xReg2 = _mm_and_si128(xReg2, mask);
        xReg1 = _mm_adds_epu8(xReg1, xReg2);
      }
      _mm_storeu_si128(parallelFD + row * backgroundCols / 16 + col, xReg1);
    }
  }
  gettimeofday(&end, NULL);
  parallelTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

  serialOutput = cropZeroes(serialOutput, padding);
  parallelOutput = cropZeroes(parallelOutput, padding);
  cv::namedWindow("serial output", cv::WINDOW_AUTOSIZE);
  cv::imshow("serial output", serialOutput);
  cv::namedWindow("parallel output", cv::WINDOW_AUTOSIZE);
  cv::imshow("parallel output", parallelOutput);
  cv::waitKey(0);

  // cv::imwrite("serialOutput.png", serialOutput);
  // cv::imwrite("parallelOutput.png", parallelOutput);

  printf("Serial time(ns): %ld\nParallel time(ns): %ld\nSpeed up: %f\n", serialTime, parallelTime,
   (float)serialTime /(float)parallelTime);
}
