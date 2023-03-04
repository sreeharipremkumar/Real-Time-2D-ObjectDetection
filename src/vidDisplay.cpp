#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/OR_functions.h"
#include <string>
#include "../include/csv_util.h"

int KNearestNeighbour(std::vector<std::pair<float, int>> &distance)
{
        int K = 3;
        std::sort(distance.begin(),distance.end());

        std::unordered_map<int, int> label_counts;
        for (int i = 0; i < K; ++i)
        {
                int label = distance[i].second;
                label_counts[label]++;
        }

        int max_count = 0, max_label = -1;
        for (const auto &label_count : label_counts)
        {
                if (label_count.second > max_count)
                {
                        max_count = label_count.second;
                        max_label = label_count.first;
                }
        }
        return max_label;
}

int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
	std::string url = "http://10.0.0.53:4747/video";
        capdev = new cv::VideoCapture(url,cv::CAP_FFMPEG);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        using namespace std;

        
        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Video Size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame,frame2,darkened,color;
        cv::Mat grassfire;
        cv::Mat label,stats,centroids,bounded;
	char c;
	char *obj_label;
	std::vector<float> feature_vec(3);
	std::vector<char *> feature_labels(10);
	std::vector<std::vector<float>> csv_data;
        std::vector<std::pair<float, int>> distance(100);

        bool Knn = false;
        bool capture = false;

        // frame = cv::imread("../Proj03Examples/test/test2.jpg",cv::IMREAD_COLOR);       //reading image
	// //test2, test4,test5
        // if(frame.empty())
        // {
        //     std::cout<<"Could not open file" << std::endl;    //throw error if file not present
        //     return 1;
        // }   
        float thresh = 81;

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if (frame.empty()) {
                  printf("frame is empty\n");
                  break;
                }

                Darken(frame,darkened);
                cv::cvtColor(darkened,frame2,6); 
                BlurThreshold(frame2,frame2,thresh);

                cv::imshow("Threshold", frame2);
                if(capture == true){
                        cv::imwrite("../Result/Threshold.jpg", frame2);
                        capture = false;
                }
                
                Grassfire(frame2,grassfire,0,6); // value1, input image, value2 - grassfire matrix
                Grassfire(frame2,grassfire,255,12); // value 3 - (0- dilation,255- erosion)
                Grassfire(frame2,grassfire,0,6); //value 4 - depth of erosion or dilation 
                cv::imshow("Grassfire", frame2);
                
                int id;
                int num_label = cv::connectedComponentsWithStats(frame2,label,stats,centroids);

		csv_data.clear();
		feature_labels.clear();
		int h = read_image_data_csv((char*)"../csv/Data.csv",feature_labels,csv_data,0);
		if(h!=-1)
		{
			id = feature_iter(feature_vec,csv_data,Knn,distance);

                        if(Knn == true)
                        {
                                id = KNearestNeighbour(distance);
                                printf("\n\n%d\n\n",id);
                        }
		}

		if(num_label>1){
			ColorDisplay(frame2, label, color);
			cv::imshow("segment", color);
                        
			LeastCentralMoments(feature_vec, frame, bounded, label, stats, centroids, num_label, feature_labels[id],Knn);
			cv::imshow("Bounding", bounded);
		}
		else
		{
			printf("No object\n");
			cv::putText(frame, "No object detected", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			cv::imshow("Bounding", frame);
		}

		char key = cv::waitKey(10);  

                if (key == 'q') {
                    break;
                }
                if (key == ','){
                        thresh = thresh + 2;
                }
                if (key == '.'){
                        thresh = thresh - 2;
                }
		if (key == 't'){
			
			
			std::cout<< "\nDo you wish to train the current Object?(y/n): ";
                        std::cin>>c;
			if(c == 'n' || c =='N')
			{
				c = '\0';
				std::cout<<"\nYou Didn't Save that Object\n";
			}
			else if(c == 'y' || c=='Y')
			{
				std::cout<<"\nEnter the Object Label : ";
				std::cin.ignore();
				std::cin>>obj_label;
				LeastCentralMoments(feature_vec, frame, bounded, label, stats, centroids, num_label, obj_label,Knn);
				append_image_data_csv((char*)"../csv/Data.csv",obj_label,feature_vec,false);
				c = '\0';
			}
		}
                if(key == 'k'){
                        if(Knn == false)
                        {
                                Knn = true;
                        }
                        else{
                                Knn = false;
                        }
                }
                if(key == 's'){
                        cv::imwrite("../Result/Image.jpg", frame);
                        cv::imwrite("../Result/Grassfire.jpg", frame2);
                        cv::imwrite("../Result/Segmented.jpg", color);
                        cv::imwrite("../Result/Classified.jpg", bounded);
                        cv::imwrite("../Result/Darken.jpg", darkened);
                        printf("Images Will be Saved");
                        capture = true;
                }
        }
        delete capdev;
        return(0);
}