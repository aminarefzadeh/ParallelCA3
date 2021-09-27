#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "x86intrin.h"
#include <sys/time.h>
#include <string>
#include <stdlib.h>

using namespace std;

#define SID_1 810195424
#define SID_2 810195578

int main() {
  printf("Amin Arefzade %d\nAmirhossein Mahmoodi %d\n", SID_1, SID_2);

  struct timeval start, end;
  long serial_time, parallel_time;

  cv::Mat first_image = cv::imread("CA03__Q1__Image__01.png", cv::IMREAD_GRAYSCALE);
  cv::Mat second_image = cv::imread("CA03__Q1__Image__02.png", cv::IMREAD_GRAYSCALE);

  unsigned int image_rows = first_image.rows;
  unsigned int image_cols = first_image.cols;

  cv::Mat serial_output (image_rows, image_cols, CV_8U);
  cv::Mat parallel_output (image_rows, image_cols, CV_8U);

  unsigned char *serial_first_image_mat = (unsigned char *) first_image.data;
  unsigned char *serial_second_image_mat = (unsigned char *) second_image.data;
  unsigned char *serial_output_mat = (unsigned char *)serial_output.data;

  gettimeofday(&start, NULL);
  for (int row = 0; row < image_rows; row++) {
    for (int col = 0; col < image_cols; col++) {
      serial_output_mat[row * image_cols + col] = abs(
        serial_first_image_mat[row * image_cols + col] - serial_second_image_mat[row * image_cols + col]
      );
    }
  }
  gettimeofday(&end, NULL);
  serial_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

  __m128i *parallel_first_image_mat = (__m128i *)first_image.data;
  __m128i *parallel_second_image_mat = (__m128i *)second_image.data;
  __m128i *parallel_output_mat = (__m128i *)parallel_output.data;
  __m128i xReg1, xReg2, res;

  gettimeofday(&start, NULL);
  for (int row = 0; row < image_rows; row++) {
    for (int col = 0; col < image_cols / 16; col++) {
      xReg1 = _mm_loadu_si128(parallel_first_image_mat + row * image_cols / 16 + col);
      xReg2 = _mm_loadu_si128(parallel_second_image_mat + row * image_cols / 16 + col);
      res = _mm_max_epu8(_mm_subs_epu8(xReg1, xReg2), _mm_subs_epu8(xReg2, xReg1));
      _mm_storeu_si128(parallel_output_mat + row * image_cols / 16 + col, res);
    }
  }
  gettimeofday(&end, NULL);
  parallel_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

  cv::namedWindow("serial output", cv::WINDOW_AUTOSIZE);
  cv::imshow("serial output", serial_output);
  cv::namedWindow("parallel output", cv::WINDOW_AUTOSIZE);
  cv::imshow("parallel output", parallel_output);
  cv::waitKey(0);

  // cv::imwrite("serial_output.png", serial_output);
  // cv::imwrite("parallel_output.png", parallel_output);

  printf("Serial time(ns): %ld\nParallel time(ns): %ld\nSpeed up: %f\n", serial_time, parallel_time,
   (float)serial_time /(float)parallel_time);
}
