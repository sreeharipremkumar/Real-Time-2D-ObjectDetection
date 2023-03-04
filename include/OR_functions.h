int BlurThreshold(cv::Mat &src, cv::Mat &dst, float thresh);
int Darken(cv::Mat &src, cv::Mat &dst);
int Grassfire(cv::Mat &src, cv::Mat &dst, int fg, int num);
int ColorDisplay(cv::Mat &src, cv::Mat &connected,cv::Mat &color);
int LeastCentralMoments(std::vector<float> &feature_vec,cv::Mat &src,cv::Mat &bounded,cv::Mat &label, cv::Mat &stats, cv::Mat &centroids,int num_label, char * &obj_label,bool knn);
int feature_iter(std::vector<float> &feature_vec,std::vector<std::vector<float>> &csv_data,bool knn,std::vector<std::pair<float, int>> &distance);
