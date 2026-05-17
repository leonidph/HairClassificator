#include <stdlib.h>
#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>

#define MLPACK_ENABLE_ANN_SERIALIZATION 
#include <mlpack.hpp>

/*#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>*/
//#include <mlpack/core/optimizers/cne/cne.hpp>
#include "csv.hpp"




        
#include <mlpack.hpp>

#if ((ENS_VERSION_MAJOR < 2) || \
    ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
  #error "need ensmallen version 2.13.0 or later"
#endif

using namespace mlpack;
//using namespace mlpack::regression;
using namespace arma;


typedef struct
{
    std::string img;
    double hairColor;
}tRecord;


void readChunk( std::vector<tRecord>& records,
                arma::mat& iIMvec,arma::mat& labels,
                size_t start,size_t cnt)
{
  
  labels.resize(1, cnt);
  iIMvec.resize(64*64, cnt);


  for(size_t i=0;i<cnt;i++)
  {
    arma::mat imageMatrix;
    mlpack::ImageOptions opts;
    
    try
    {      
      bool success = mlpack::data::Load(records[start+i].img, imageMatrix,opts, false );      
      if(!success)
      {
          std::cerr << "Error: Could not load image " << records[start+i].img << std::endl;
          exit(1);
      }
     
      imageMatrix = imageMatrix.reshape(64, 64);
      labels(0,i) = records[start+i].hairColor;
    }
    catch(const std::exception& e)
    {
      std::cerr << "Failed  on "<<start+i << '\n';
      std::cerr << e.what() << '\n';
    } 
    iIMvec.col(i)     =imageMatrix.reshape( 64*64,1);
  }
}

void readSingle( tRecord& record,
                arma::mat& iIMvec,arma::mat& labels,
                size_t id)
{
  
  labels.resize(1, 1);
  iIMvec.resize(64*64, 1);


  
    arma::mat imageMatrix;
    mlpack::ImageOptions opts;
    
    try
    {      
      bool success = mlpack::data::Load(record.img, imageMatrix,opts, false );      
      if(!success)
      {
          std::cerr << "Error: Could not load image " << record.img << std::endl;
          exit(1);
      }
     
      imageMatrix = imageMatrix.reshape(64, 64);
      labels(0,0) = record.hairColor;
    }
    catch(const std::exception& e)
    {
      std::cerr << "Failed  on "<<id << '\n';
      std::cerr << e.what() << '\n';
    } 
    iIMvec.col(0)     =imageMatrix.reshape( 64*64,1);
  
}

/*
mlpack::ann::FFN<mlpack::ann::NegativeLogLikelihood<>>& createNetwork()
{
  mlpack::ann::FFN<mlpack::ann::NegativeLogLikelihood<> > network;
  network.Add<mlpack::ann::Linear<> >(784, 784);
  network.Add<mlpack::ann::SigmoidLayer<> >();

  network.Add<mlpack::ann::Linear<> >(784, 10);
  network.Add<mlpack::ann::LogSoftMax<> >(); 
  return network;
}*/

int main()
{
  //const size_t points = 500;
  const size_t chunkSize = 20000;
   int stat[5]={0};
  
    CSVReader csv;
    std::vector<tRecord> records;
    // Read CSV file
    if (csv.readFile("/home/leonidp/ds/list_attr_celeba.csv"))
     {
        std::cout << "CSV file loaded successfully!" << std::endl;
        std::cout << "Total rows: " << csv.getRowCount() << std::endl << std::endl;

        // Print all data
        //csv.printData();
        /*for(int i=0; i<csv.getColumnCount(0); ++i)
        {
            std::cout << "Column " << i << ": " << csv.getCell(0, i) << std::endl;
        }
        exit(0);  */
        // Access specific cell
       
        for(int i=1; i<csv.getRowCount(); ++i)
        {
            char filename[100];
            sprintf(filename, "/home/leonidp/ds/img_align_celeba/img_align_celeba/%06d.jpg", csv.getCellAsInt ( i,0) );
                tRecord rec;
                rec.img = filename;
                if(csv.getCellAsInt ( i, 9)==1)
                {
                    rec.hairColor = .1;
                    stat[1]++;
                }
                else if(csv.getCellAsInt ( i, 10)==1)
                {
                    rec.hairColor = .4;
                    stat[2]++;
                }
                else if(csv.getCellAsInt ( i, 12)==1)
                  { 
                     rec.hairColor = .6;
                     stat[3]++; 
                  }
                else if(csv.getCellAsInt ( i, 18)==1)
                    {
                      rec.hairColor = .8;
                      stat[4]++;
                    }
                else
                {
                  continue;
                  //rec.hairColor = 0;
                  //stat[0]++;
                }

                records.push_back(rec);
        }
     }
     for(int i=0;i<5;i++)
     {
       std::cout << "Stat " << i << ": " << stat[i] << std::endl;
     }
    std::cout << "Record = " << records.size() << std::endl;
    int  ImageSize = 768;

    ens::Adam optimizer(chunkSize, chunkSize);
// MaxIterations = epochs * (num_samples / batch_size)
    optimizer.MaxIterations() = 200 * (records.size() / chunkSize);
    FFN<NegativeLogLikelihood> network;

    if(!data::Load("hair_classificator_model.bin",network))
    {
      std::cout << "No saved network  - going to learn it! " << std::endl;
      network.Add<Linear >(ImageSize);
      network.Add<Sigmoid>();
      network.Add<Linear>(ImageSize/4);
      network.Add<Sigmoid>(); 
      network.Add<Linear>(10);
      network.Add<Linear>(ImageSize/8);
      network.Add<Sigmoid>(); 
      network.Add<Linear>(ImageSize/16);
      network.Add<Sigmoid>();             
      network.Add<Sigmoid>(); 
      network.Add<Linear>(1);
      network.Add<LogSoftMax>(); 
      
      

      arma::mat iIMvec;
      arma::mat  labels;
      
      for(size_t i=0;i<records.size();i=i+chunkSize)
      {
            std::cout << "Processing chunk starting at index " << i << std::endl;
            readChunk(records, iIMvec,labels,i,std::min(chunkSize, records.size() - i));
            double lost = network.Train(iIMvec, labels,optimizer, ens::ProgressBar(),ens::PrintLoss() );
            std::cout<< std::endl<< std::endl<< "Chunk processed with loss: " << lost << std::endl;
            data::Save("hair_classificator_model.bin","HairColor", network);
      }
    }
    else
    {

      std::cout << "Ok ! We have it , lets check."<< std::endl; 
      arma::mat iIMvec;
      arma::mat  labels;
      for(int i=0;i<100;i++)
      {
          size_t id = rand() % records.size();
          readSingle( records[id], iIMvec,labels,id);

          arma::mat predictions;
          predictions.resize(1,1);
          network.Predict(iIMvec, predictions);
          std::cout << "Predictions: " << predictions.col(0).row(0) << std::endl;
          std::cout << "Actual: " << labels.col(0).t() << std::endl;
      }      
    }
    std::cout << "Done !"  << std::endl;

    return 0;
}
