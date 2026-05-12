#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/data/split_data.hpp>

using namespace mlpack;
using namespace mlpack::regression;
using namespace arma;

int main()
{
  const size_t points = 500;
  arma::mat dataset = arma::randn<arma::mat>(2, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points; ++i)
  {
    labels[i] = (dataset(0, i) + dataset(1, i) > 0.0) ? 1 : 0;
  }

  arma::mat trainData;
  arma::mat testData;
  arma::Row<size_t> trainLabels;
  arma::Row<size_t> testLabels;
  data::Split(dataset, labels, trainData, testData, trainLabels, testLabels, 0.3);

  const double lambda = 0.001;
  LogisticRegression<> model(trainData, trainLabels, lambda);

  arma::Row<size_t> predictions;
  model.Classify(testData, predictions);

  double accuracy = arma::mean(arma::conv_to<arma::rowvec>::from(predictions == testLabels)) * 100.0;
  std::cout << "Test accuracy: " << accuracy << "%\n";

  cv::Mat image(300, 300, CV_8UC3, cv::Scalar(50, 150, 200));
  cv::circle(image, cv::Point(150, 150), 80, cv::Scalar(0, 255, 0), -1);
  cv::putText(image, "mlpack + OpenCV", cv::Point(30, 170), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
  cv::imwrite("output.png", image);
  std::cout << "Wrote output.png with OpenCV sample image." << std::endl;

  return 0;
}
