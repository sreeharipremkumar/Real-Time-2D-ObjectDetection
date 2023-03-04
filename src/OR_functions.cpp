#include <opencv2/highgui.hpp> //including needed files
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>
#include <cmath>
#include "../include/csv_util.h"

int Darken(cv::Mat &src, cv::Mat &dst)
{

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    for (int i = 0; i < hsv.rows; i++)
    {
        cv::Vec3b *sptr = hsv.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hsv.cols; j++)
        {
            if (sptr[j][1] > 120)
            {
                sptr[j][2] = sptr[j][2] / 4;
            }
        }
    }
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
    return 0;
}

int BlurThreshold(cv::Mat &src, cv::Mat &dst, float thresh)
{

    cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0);
    for (int i = 0; i < src.rows; i++)
    {
        uchar *sptr = src.ptr<uchar>(i);
        uchar *dptr = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; j++)
        {
            if (sptr[j] > thresh)
            {
                dptr[j] = 0;
            }
            else
            {
                dptr[j] = 255;
            }
        }
    }
    return 0;
}

int dilation_erotion(cv::Mat &src, cv::Mat &grassfire, int num, int fgbg)
{
    for (int i = 0; i < src.rows; i++)
    {
        int *dptr = grassfire.ptr<int>(i);
        uchar *sptr = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++)
        {
            if (dptr[j] < num && dptr[j] != 0)
            {
                sptr[j] = 255 - fgbg;
            }
        }
    }
    return 0;
}

int Grassfire(cv::Mat &src, cv::Mat &grassfire, int fg, int num)
{

    grassfire = cv::Mat::zeros(src.size(), CV_32S);
    int temp1, temp2;
    for (int i = 0; i < src.rows; i++)
    {
        int *dptr = grassfire.ptr<int>(i);
        int *dptrm1 = grassfire.ptr<int>(i - 1);

        uchar *sptr = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++)
        {
            if (sptr[j] == fg)
            {
                if (i == 0)
                {
                    temp1 = 255 - fg;
                }
                else
                {
                    temp1 = dptrm1[j];
                }
                if (j == 0)
                {
                    temp2 = 255 - fg;
                }
                else
                {
                    temp2 = dptr[j - 1];
                }
                dptr[j] = std::min(temp2, temp1) + 1;
            }
        }
    }

    for (int i = src.rows - 1; i >= 0; i--)
    {
        int *dptr = grassfire.ptr<int>(i);
        int *dptrp1 = grassfire.ptr<int>(i + 1);
        uchar *sptr = src.ptr<uchar>(i);

        for (int j = src.cols - 1; j >= 0; j--)
        {
            if (sptr[j] == fg)
            {
                if (i == src.rows - 1)
                {
                    temp1 = 255 - fg;
                }
                else
                {
                    temp1 = dptrp1[j];
                }
                if (j == src.cols - 1)
                {
                    temp2 = 255 - fg;
                }
                else
                {
                    temp2 = dptr[j + 1];
                }
                dptr[j] = std::min(dptr[j] + 0, std::min(temp2, temp1) + 1);
            }
        }
    }
    dilation_erotion(src, grassfire, num, fg);
    return 0;
}

int ColorDisplay(cv::Mat &src, cv::Mat &connected, cv::Mat &color)
{
    uchar RGB[10][3] = {
        {255, 182, 193}, // Light Pink
        {152, 255, 152}, // Mint Green
        {245, 245, 220}, // Beige
        {230, 230, 250}, // Lavender
        {255, 0, 255},   // Magenta
        {0, 255, 255},   // Cyan
        {128, 0, 0},     // Maroon
        {0, 128, 0},     // Olive
        {0, 0, 128},     // Navy
        {135, 206, 235}  // Sky Blue
    };
    // cvtColor(src, color, cv::COLOR_GRAY2BGR); // convert to color image

    // set color for each pixel
    color = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int i = 0; i < color.rows; i++)
    {
        cv::Vec3b *cptr = color.ptr<cv::Vec3b>(i);
        int *segment = connected.ptr<int>(i);
        for (int j = 0; j < color.cols; j++)
        {
            if (segment[j] < 10)
            {
                cptr[j][0] = RGB[segment[j]][2];
                cptr[j][1] = RGB[segment[j]][1];
                cptr[j][2] = RGB[segment[j]][0];
            }
            else
            {
                cptr[j][0] = 0;
                cptr[j][1] = 0;
                cptr[j][2] = 0;
            }
        }
    }
    return 0;
}

std::tuple<std::vector<float>,std::vector<float>,std::vector<float>> CalcMeans(std::vector<std::vector<float>> &csv_data) 
{
    std::vector<float> means(csv_data[0].size());
    std::vector<float> max,min;
    max.assign(csv_data[0].size(),-std::numeric_limits<float>::infinity());
    min.assign(csv_data[0].size(),std::numeric_limits<float>::infinity());

    for (int col = 0; col < csv_data[0].size(); col++) {
        float sum = 0.0;
        for (int row = 0; row < csv_data.size(); row++) {
            
            sum += csv_data[row][col];

            if(min[col] > csv_data[row][col])
            {
                min[col] = csv_data[row][col];
            }
            if(max[col] < csv_data[row][col])
            {
                max[col] = csv_data[row][col];
            }
        }
        means[col] = sum / csv_data.size();
    }
    return std::make_tuple(means, max, min);
}

std::vector<float> CalcStdDevs(const std::vector<std::vector<float>>& csv_data,std::vector<float> &means) {
    
    std::vector<float> std_devs(csv_data[0].size());
    for (int col = 0; col < csv_data[0].size(); col++) {
        float sum = 0.0;
        for (int row = 0; row < csv_data.size(); row++) {
            sum += std::pow(csv_data[row][col] - means[col], 2.0);
        }
        std_devs[col] = std::sqrt(sum / (csv_data.size() - 1));
    }
    return std_devs;
}

float SSE(std::vector<float> &Ft, std::vector<float> &Fi,std::vector<float> &mean,std::vector<float> &std_dev,std::vector<float> &max,std::vector<float> &min)
{
    float sum = 0;int mul = 1;

    for (int i = 0; i < Fi.size(); i++)
    {
        float val1 = (Ft[i]-mean[i])/std_dev[i];
        val1 = (Ft[i] - min[i])/(max[i]-min[i]);
        float val2 = (Fi[i]-mean[i])/std_dev[i];
        val2 = (Fi[i] - min[i])/(max[i]-min[i]);
        if(i==2){
            mul = 0.85;
        }
        else if(i==1){
            mul = 0.9;
        }
        else{
            mul = 1;
        }
        sum = sum + mul*(val1-val2) * (val1 - val2);
        //printf("\ni= %d  Ft=%f  Fi=%f val1 = %f      Val2 = %f     sum = %f",i,Ft[i],Fi[i],val1,val2,sum);
    }
    return sum;
}


int feature_iter(std::vector<float> &feature_vec,std::vector<std::vector<float>> &csv_data,bool knn,std::vector<std::pair<float, int>> &distance)
{   
    std::vector<float> max,min,means;
    std::tie(means,max,min) = CalcMeans(csv_data);
    std::vector<float> std_devs = CalcStdDevs(csv_data,means);
    

    float sum,least = std::numeric_limits<float>::infinity(),id;
    for(int i=0;i<csv_data.size();i++)
    {
        sum = SSE(feature_vec,csv_data[i],means,std_devs,max,min);

        if(knn == true)
        {
            distance[i].first =sum;
            distance[i].second = i;
        }
        //printf("\nmean = %f   std_dev = %f  sum = %f i=%d\n",means[i],std_devs[i],sum,i);
        if (sum<least)
        {
            least = sum;
            id = i;
        }
    }
    return id;
}



int LeastCentralMoments(std::vector<float> &feature_vec,cv::Mat &src, cv::Mat &bounding, cv::Mat &label, cv::Mat &stats, cv::Mat &centroids, int num_label,char * &obj_label,bool knn)
{
    bounding = src.clone();

    if(obj_label == NULL){
        return 0;
    }
    const double pi = 3.1415926;

    std::vector<int> largest(num_label - 1);
    std::iota(largest.begin(), largest.end(), 1);
    std::sort(largest.begin(), largest.end(), [&](int a, int b)
              { return stats.at<int>(a, cv::CC_STAT_AREA) > stats.at<int>(b, cv::CC_STAT_AREA); });

    int id = largest.at(0);
    int Mnum = stats.at<int>(id, cv::CC_STAT_AREA);
    double X_centroid = centroids.at<double>(id, 0);
    double Y_centroid = centroids.at<double>(id, 1);

    double central_y_moment = 0, central_x_moment = 0, cross_moment = 0, alpha;
    for (int y = 0; y < label.rows; y++)
    {
        int *seg = label.ptr<int>(y);
        for (int x = 0; x < label.cols; x++)
        {
            if (seg[x] == id)
            {
                central_y_moment = central_y_moment + (y - Y_centroid) * (y - Y_centroid) / Mnum;
                central_x_moment = central_x_moment + (x - X_centroid) * (x - X_centroid) / Mnum;
                cross_moment = cross_moment + (x - X_centroid) * (y - Y_centroid) / Mnum;
            }
        }
    }
    alpha = 0.5 * std::atan2(2 * cross_moment, central_x_moment - central_y_moment);

    float sin_alpha = std::sin(alpha + pi);
    float cos_alpha = std::cos(alpha + pi);
    float oriented_central_moment = 0;
    for (int y = 0; y < label.rows; y++)
    {
        int *seg = label.ptr<int>(y);
        for (int x = 0; x < label.cols; x++)
        {
            if (seg[x] == id)
            {
                oriented_central_moment = oriented_central_moment + (((y - Y_centroid)*cos_alpha  + (x - X_centroid)* sin_alpha)*((y - Y_centroid)*cos_alpha + (x - X_centroid)*sin_alpha));
            }
        }
    }

    float width, height;

    if (alpha > 0.1 || alpha < -0.1)
    {
        float slope_long = std::tan(alpha);
        float slope_short = -1 / slope_long;

        float C1 = Y_centroid - slope_long * X_centroid; // intercept given slope and point
        float C2 = Y_centroid - slope_short * X_centroid;
        float left_x = -1, left_y = -1, right_x = -1, right_y = -1;
        float bot_x = -1, bot_y = -1,top_x = -1, top_y = -1;

        for (int x = 0; x < src.cols; x++)
        {
            float y_long = slope_long * x + C1;
            float y_short = slope_short*x + C2;

            int x_int = cvRound(x);
            int yl_int = cvRound(y_long);
            int ys_int = cvRound(y_short);
            if (x_int > 0 && x_int < src.cols && yl_int > 0 && yl_int < src.rows)
            {
                if (label.at<int>(yl_int, x_int) == id)
                {
                    if (left_x == -1 && left_y == -1)
                    {
                        left_x = x_int;
                        left_y = yl_int;
                    }
                    right_x = x_int;
                    right_y = yl_int;
                    cv::circle(bounding, cv::Point(x_int, yl_int), 1, cv::Scalar(0, 0, 255), -1);
                }
            }
            if (x_int > 0 && x_int < src.cols && ys_int > 0 && ys_int < src.rows)
            {
                if (label.at<int>(ys_int, x_int) == id)
                {
                    if (bot_x == -1 && bot_y == -1)
                    {
                        bot_x = x_int;
                        bot_y = ys_int;
                    }
                    top_x = x_int;
                    top_y = ys_int;
                    cv::circle(bounding, cv::Point(x_int, ys_int), 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
        width = sqrt((right_x - left_x) * (right_x - left_x) + (right_y - left_y) * (right_y - left_y));
        height = sqrt((top_x - bot_x) * (top_x - bot_x) + (top_y - bot_y) * (top_y - bot_y));
    }
    else
    {
        width = stats.at<int>(id, cv::CC_STAT_WIDTH);
        height = stats.at<int>(id, cv::CC_STAT_HEIGHT);
    }

    cv::Point2f center(X_centroid, Y_centroid);

    // Create a Size2f object representing the width and height of the bounding box
    cv::Size2f size(width,height);
    cv::RotatedRect boundingbox(center, size, alpha*pi/180);
    cv::Scalar color(0, 255, 0); // Green color

    cv::Point2f corners[4];
    
    boundingbox.points(corners);

    for (int i = 0; i < 4; i++)
    {
        cv::Point2f corner = corners[i];
        corners[i].x = center.x + (corner.x - center.x) * cos(alpha) - (corner.y - center.y) * sin(alpha);
        corners[i].y = center.y + (corner.x - center.x) * sin(alpha) + (corner.y - center.y) * cos(alpha);
    }

    // Draw the bounding box
    cv::line(bounding, corners[0], corners[1], color, 2);
    cv::line(bounding, corners[1], corners[2], color, 2);
    cv::line(bounding, corners[2], corners[3], color, 2);
    cv::line(bounding, corners[3], corners[0], color, 2);

    std::string text(obj_label);

    if(knn == true){
        text = text.substr(0, text.length() - 1);
        cv::putText(bounding, "KNN Mode", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			
    }
    cv::putText(bounding, text, cv::Point(center.x + boundingbox.size.width / 2, center.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

    float ratio = height/width;
    float percentage_filled = stats.at<int>(id,cv::CC_STAT_AREA)/(height*width);
    feature_vec[0] = ratio;
    feature_vec[1] = percentage_filled;
    feature_vec[2] = oriented_central_moment;
    return 0;
}
