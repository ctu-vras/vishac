#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include "shape_completion/smooth_pcl.h"
#include "ros/ros.h"

bool smooth(shape_completion::smooth_pcl::Request &req,
                 shape_completion::smooth_pcl::Response &res)
{

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);

    std::vector<double> points = req.points;
    cloud->width = points.size()/4;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->resize(cloud->width*cloud->height);
    #pragma omp parallel for
        for (int point_id=0; point_id < points.size()/4; point_id++)
        {
          cloud->at(point_id).x = points[point_id*4];
          cloud->at(point_id).y = points[(point_id*4)+1];
          cloud->at(point_id).z = points[(point_id*4)+2];
          cloud->at(point_id).rgba = points[(point_id*4)+3];
        }

    if (req.method!="arm_keep") {
        pcl::VoxelGrid<pcl::PointXYZRGBA> avg;
        double leaf_size = req.leaf_size;

        avg.setInputCloud(cloud);
        avg.setLeafSize(leaf_size, leaf_size, leaf_size);
        avg.filter(*cloud_filtered);
    }
    else {cloud_filtered = cloud;}

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGBA>);
    if (req.method != "arm")
    {
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr search(new pcl::search::KdTree<pcl::PointXYZRGBA>);
        pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA> mls;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth (new pcl::PointCloud<pcl::PointXYZRGBA>);
        search->setInputCloud(cloud_filtered);

        mls.setInputCloud(cloud_filtered);
        mls.setPolynomialFit(true);
        mls.setPolynomialOrder(2);//General 2-5
        mls.setSearchMethod(search);
        mls.setSearchRadius(req.smooth_factor);//The bigger the smoother the greater
        mls.process(*smooth);

        if (req.method != "arm_keep"){
            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
            sor.setInputCloud (smooth);
            sor.setMeanK (100);
            sor.setStddevMulThresh (2.5);
            sor.filter (*result);
        }
        else{result = smooth;}
    }
    else {result = cloud_filtered;}

    std::vector<double> points_out{};
    #pragma omp parallel for
        for(auto& point : result->points)
        {
            points_out.push_back(point.x);
            points_out.push_back(point.y);
            points_out.push_back(point.z);
            points_out.push_back(point.rgba);
        }
    res.points = points_out;

  return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "smooth_pcl_server");

  ros::NodeHandle node_handle;
  ros::ServiceServer service = node_handle.advertiseService("smooth_pcl", smooth);
  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}