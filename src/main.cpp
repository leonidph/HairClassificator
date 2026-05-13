#include <stdlib.h>
#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/data/split_data.hpp>
#include "csv.hpp"

using namespace mlpack;
//using namespace mlpack::regression;
using namespace arma;


typedef struct
{
    std::string img;
    int hairColor;
}tRecord;


int main()
{
  const size_t points = 500;
  
  std::cout << "Wrote output.png with OpenCV sample image." << std::endl;
 CSVReader csv;
    std::vector<tRecord> records;
    // Read CSV file
    if (csv.readFile("/home/leonidp/ds/list_attr_celeba.csv"))
     {
        std::cout << "CSV file loaded successfully!" << std::endl;
        std::cout << "Total rows: " << csv.getRowCount() << std::endl << std::endl;

        // Print all data
        //csv.printData();

        // Access specific cell
        for(int i=1; i<csv.getRowCount(); ++i)
        {
            char filename[10];
            sprintf(filename, "%06d.jpg", csv.getCellAsInt ( i,0) );
                tRecord rec;
                rec.img = filename;
                if(csv.getCellAsInt ( i, 9))
                    rec.hairColor = 1;
                else if(csv.getCellAsInt ( i, 10))
                    rec.hairColor = 2;
                else if(csv.getCellAsInt ( i, 12))
                    rec.hairColor = 3;
                else if(csv.getCellAsInt ( i, 18))
                    rec.hairColor = 4;
                else
                rec.hairColor = 0;

                records.push_back(rec);
        }
     }
    std::cout << "Record = " << records.size() << std::endl;

    
  return 0;
}
