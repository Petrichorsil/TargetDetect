// RGB-LiDAR detection node without experiment recording.
// Synchronization, YOLO segmentation, projection, clustering and the existing
// four point-cloud topics are retained. No timestamp whitelist or result export.
// The original class, ROS node name and topic names remain for compatibility.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/bind.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

// ==========================================================================
// USER CONFIGURATION
// Edit this block when the topics or detection settings change.
// These defaults allow launching without private ROS parameters.
// ==========================================================================
namespace defaults {
constexpr const char* kImageTopic = "/usb_cam_before/image_raw/compressed";
constexpr const char* kCloudTopic = "/QT128/lidar_points";
constexpr const char* kYoloHost = "127.0.0.1";
constexpr int kYoloPort = 65432;

constexpr bool kPublishClouds = true;
constexpr double kCropHalfExtent = 50.0;
constexpr double kLidarHeight = 3.5;
constexpr double kObstacleHeightThreshold = 0.05;
constexpr double kMinCameraDepth = 0.1;
constexpr double kSyncTolerance = 0.06;
constexpr double kClusterRadius = 0.5;
constexpr int kMinClusterSize = 20;
constexpr int kMaxClusterSize = 25000;
}  // namespace defaults

struct YOLOSegmentationResult {
    int class_id{-1};
    std::vector<cv::Point> polygon;
    cv::Rect bounding_box;
};

struct DetectionResponse {
    bool communication_ok{false};
    std::vector<YOLOSegmentationResult> detections;
};

struct ProjectedPoint {
    PointT lidar;
    int u{0};                      // image column
    int v{0};                      // image row
    int instance_id{-1};           // YOLO instance after overlap resolution
    int class_id{-1};
};

using Cluster = std::vector<std::size_t>;  // indices into std::vector<ProjectedPoint>

class ExperimentRecorder {
public:
    explicit ExperimentRecorder(ros::NodeHandle& nh);
    ~ExperimentRecorder();

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::CompressedImage, sensor_msgs::PointCloud2>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void syncCallback(const sensor_msgs::CompressedImageConstPtr& image_msg,
                      const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

    DetectionResponse detectObjectsWithYOLO(const cv::Mat& image,
                                            const std::string& frame_key);
    std::vector<YOLOSegmentationResult> parseYOLOResponse(
        const std::string& response) const;
    bool connectToYOLOServer();
    void disconnectFromYOLOServer();

    std::vector<Cluster> clusterSubset(
        const std::vector<ProjectedPoint>& points,
        const std::vector<std::size_t>& subset,
        double radius,
        int min_cluster_size,
        int max_cluster_size) const;

    void processFrame(const cv::Mat& image,
                      const PointCloudT::Ptr& cloud,
                      const ros::Time& cloud_stamp);

    void publishMethod(const std::vector<ProjectedPoint>& points,
                       const std::vector<uint8_t>& selected,
                       const ros::Time& stamp,
                       ros::Publisher& publisher) const;

    static std::string frameKey(const ros::Time& cloud_stamp);
    static void markClusters(const std::vector<Cluster>& clusters,
                             std::vector<uint8_t>& flags,
                             bool largest_only);
    static Eigen::Isometry3d poseFromRPY(double x, double y, double z,
                                         double roll_deg, double pitch_deg,
                                         double yaw_deg);

    ros::NodeHandle& nh_;
    message_filters::Subscriber<sensor_msgs::CompressedImage> image_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    boost::shared_ptr<Synchronizer> sync_;

    ros::Publisher projection_pub_;
    ros::Publisher geometry_pub_;
    ros::Publisher largest_pub_;
    ros::Publisher proposed_pub_;

    std::mutex callback_mutex_;
    int yolo_socket_{-1};
    bool yolo_connected_{false};

    std::string image_topic_;
    std::string cloud_topic_;
    std::string yolo_host_;
    int yolo_port_{65432};
    bool publish_clouds_{true};

    double crop_half_extent_{50.0};
    double lidar_height_{3.5};
    double obstacle_height_threshold_{0.05};
    double min_camera_depth_{0.1};
    double sync_tolerance_{0.06};
    double default_radius_{0.5};
    int default_min_cluster_size_{20};
    int max_cluster_size_{25000};

    Eigen::Matrix3d intrinsics_;
    Eigen::Matrix4d camera_body_from_lidar_;
    Eigen::Isometry3d base_from_lidar_;
};

ExperimentRecorder::ExperimentRecorder(ros::NodeHandle& nh) : nh_(nh) {
    // Intentionally assign the source-code constants directly.  Do not read
    // private ROS parameters here: rosrun/roslaunch parameters can remain on
    // the ROS parameter server after an earlier process exits and silently
    // override a newly edited default path.
    image_topic_ = defaults::kImageTopic;
    cloud_topic_ = defaults::kCloudTopic;
    yolo_host_ = defaults::kYoloHost;
    yolo_port_ = defaults::kYoloPort;
    publish_clouds_ = defaults::kPublishClouds;
    crop_half_extent_ = defaults::kCropHalfExtent;
    lidar_height_ = defaults::kLidarHeight;
    obstacle_height_threshold_ = defaults::kObstacleHeightThreshold;
    min_camera_depth_ = defaults::kMinCameraDepth;
    sync_tolerance_ = defaults::kSyncTolerance;
    default_radius_ = defaults::kClusterRadius;
    default_min_cluster_size_ = defaults::kMinClusterSize;
    max_cluster_size_ = defaults::kMaxClusterSize;

    if (crop_half_extent_ <= 0.0 || lidar_height_ <= 0.0 ||
        min_camera_depth_ <= 0.0 || sync_tolerance_ <= 0.0 ||
        default_radius_ <= 0.0 || default_min_cluster_size_ <= 0 ||
        max_cluster_size_ < default_min_cluster_size_) {
        throw std::invalid_argument(
            "Invalid detection parameter: extents, depths, tolerances and "
            "radii must be positive, and max_cluster_size must be no smaller "
            "than min_cluster_size.");
    }

    // Camera intrinsics and LiDAR-to-camera-body transform are copied from the
    // verified projection code. X is forward depth, Y is lateral, Z is vertical.
    intrinsics_ << 1140.617714754220, 0.0, 1001.227213560750,
                   0.0, 1139.690154300230, 587.831470142597,
                   0.0, 0.0, 1.0;
    camera_body_from_lidar_ <<
        0.999176,  -0.0395497,  0.00916086, 0.162759,
        0.0391545,  0.998436,   0.0399048,  0.253779,
       -0.0107248, -0.0395133,  0.999161,   0.295625,
        0.0,        0.0,        0.0,        1.0;
    base_from_lidar_ = poseFromRPY(2.2, 0.02, 2.8582, 0.0, 0.0, 0.0);

    image_sub_.subscribe(nh_, image_topic_, 100);
    cloud_sub_.subscribe(nh_, cloud_topic_, 100);
    sync_.reset(new Synchronizer(SyncPolicy(100)));
    sync_->connectInput(image_sub_, cloud_sub_);
    sync_->setMaxIntervalDuration(ros::Duration(sync_tolerance_));
    sync_->registerCallback(boost::bind(&ExperimentRecorder::syncCallback,
                                        this, _1, _2));

    projection_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/experiment/projection_only", 1);
    geometry_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/experiment/geometry_only", 1);
    largest_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/experiment/largest_cluster", 1);
    proposed_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/experiment/proposed", 1);

    connectToYOLOServer();
    ROS_INFO("RGB-LiDAR detector ready; experiment recording disabled.");
}

ExperimentRecorder::~ExperimentRecorder() { disconnectFromYOLOServer(); }

Eigen::Isometry3d ExperimentRecorder::poseFromRPY(
    double x, double y, double z, double roll_deg, double pitch_deg,
    double yaw_deg) {
    const double d2r = M_PI / 180.0;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() << x, y, z;
    pose.rotate(Eigen::AngleAxisd(yaw_deg * d2r, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_deg * d2r, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_deg * d2r, Eigen::Vector3d::UnitX()));
    return pose;
}

std::string ExperimentRecorder::frameKey(const ros::Time& stamp) {
    std::ostringstream ss;
    ss << "cloud_" << stamp.sec << "_" << std::setw(9) << std::setfill('0')
       << stamp.nsec;
    return ss.str();
}

void ExperimentRecorder::syncCallback(
    const sensor_msgs::CompressedImageConstPtr& image_msg,
    const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
        if (!cv_ptr || cv_ptr->image.empty()) {
            ROS_WARN("Empty decoded RGB image; synchronized pair skipped.");
            return;
        }
        PointCloudT::Ptr cloud(new PointCloudT);
        pcl::fromROSMsg(*cloud_msg, *cloud);
        if (cloud->empty()) {
            ROS_WARN("Empty LiDAR cloud; synchronized pair skipped.");
            return;
        }
        processFrame(cv_ptr->image, cloud, cloud_msg->header.stamp);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge error: %s", e.what());
    } catch (const std::exception& e) {
        ROS_ERROR("Detection frame failed: %s", e.what());
    }
}

bool ExperimentRecorder::connectToYOLOServer() {
    if (yolo_connected_) return true;
    yolo_socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (yolo_socket_ < 0) {
        ROS_ERROR("Could not create YOLO socket.");
        return false;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(yolo_port_);
    if (inet_pton(AF_INET, yolo_host_.c_str(), &address.sin_addr) <= 0 ||
        connect(yolo_socket_, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) < 0) {
        ROS_WARN("Could not connect to YOLO server at %s:%d.",
                 yolo_host_.c_str(), yolo_port_);
        ::close(yolo_socket_);
        yolo_socket_ = -1;
        return false;
    }
    timeval timeout{};
    timeout.tv_sec = 5;
    setsockopt(yolo_socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout,
               sizeof(timeout));
    yolo_connected_ = true;
    return true;
}

void ExperimentRecorder::disconnectFromYOLOServer() {
    if (yolo_socket_ >= 0) ::close(yolo_socket_);
    yolo_socket_ = -1;
    yolo_connected_ = false;
}

std::vector<YOLOSegmentationResult> ExperimentRecorder::parseYOLOResponse(
    const std::string& response) const {
    std::vector<YOLOSegmentationResult> results;
    std::istringstream stream(response);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line == "END") continue;
        if (line.rfind("ERROR:", 0) == 0) continue;
        std::vector<int> values;
        std::istringstream line_stream(line);
        std::string token;
        while (std::getline(line_stream, token, ',')) {
            try {
                values.push_back(std::stoi(token));
            } catch (...) {
                values.clear();
                break;
            }
        }
        if (values.size() < 7 || values.size() % 2 == 0) continue;
        YOLOSegmentationResult result;
        result.class_id = values.front();
        for (std::size_t i = 1; i + 1 < values.size(); i += 2) {
            result.polygon.emplace_back(values[i], values[i + 1]);
        }
        if (result.polygon.size() >= 3) {
            result.bounding_box = cv::boundingRect(result.polygon);
            results.push_back(std::move(result));
        }
    }
    return results;
}

DetectionResponse ExperimentRecorder::detectObjectsWithYOLO(
    const cv::Mat& image, const std::string& frame_key) {
    DetectionResponse output;
    if (!yolo_connected_ && !connectToYOLOServer()) return output;

    const fs::path temp_path = fs::path("/tmp") / (frame_key + ".jpg");
    if (!cv::imwrite(temp_path.string(), image)) return output;

    const std::string request = temp_path.string();
    if (::send(yolo_socket_, request.data(), request.size(), MSG_NOSIGNAL) < 0) {
        disconnectFromYOLOServer();
        fs::remove(temp_path);
        return output;
    }

    std::string response;
    char buffer[4096];
    while (response.find("END\n") == std::string::npos) {
        const ssize_t received = ::recv(yolo_socket_, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            disconnectFromYOLOServer();
            fs::remove(temp_path);
            return output;
        }
        response.append(buffer, static_cast<std::size_t>(received));
    }
    fs::remove(temp_path);
    if (response.rfind("ERROR:", 0) == 0) return output;
    output.communication_ok = true;
    output.detections = parseYOLOResponse(response);
    return output;
}

std::vector<Cluster> ExperimentRecorder::clusterSubset(
    const std::vector<ProjectedPoint>& points,
    const std::vector<std::size_t>& subset,
    double radius,
    int min_cluster_size,
    int max_cluster_size) const {
    std::vector<Cluster> result;
    if (subset.size() < static_cast<std::size_t>(min_cluster_size)) return result;

    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->reserve(subset.size());
    for (const std::size_t index : subset) cloud->push_back(points[index].lidar);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<PointT> extractor;
    extractor.setClusterTolerance(radius);
    extractor.setMinClusterSize(min_cluster_size);
    extractor.setMaxClusterSize(max_cluster_size);
    extractor.setSearchMethod(tree);
    extractor.setInputCloud(cloud);

    std::vector<pcl::PointIndices> pcl_clusters;
    extractor.extract(pcl_clusters);
    for (const auto& pcl_cluster : pcl_clusters) {
        Cluster mapped;
        mapped.reserve(pcl_cluster.indices.size());
        for (const int local_index : pcl_cluster.indices) {
            mapped.push_back(subset.at(static_cast<std::size_t>(local_index)));
        }
        result.push_back(std::move(mapped));
    }
    std::sort(result.begin(), result.end(),
              [](const Cluster& a, const Cluster& b) {
                  return a.size() > b.size();
              });
    return result;
}

void ExperimentRecorder::markClusters(const std::vector<Cluster>& clusters,
                                      std::vector<uint8_t>& flags,
                                      bool largest_only) {
    if (clusters.empty()) return;
    const std::size_t count = largest_only ? 1 : clusters.size();
    for (std::size_t c = 0; c < count; ++c) {
        for (const std::size_t point_index : clusters[c]) flags[point_index] = 1;
    }
}

void ExperimentRecorder::processFrame(const cv::Mat& image,
                                      const PointCloudT::Ptr& cloud,
                                      const ros::Time& cloud_stamp) {
    const std::string key = frameKey(cloud_stamp);
    const DetectionResponse yolo = detectObjectsWithYOLO(image, key);
    if (!yolo.communication_ok) {
        ROS_WARN_STREAM("YOLO communication failed; frame skipped: " << key);
        return;
    }

    cv::Mat instance_map(image.rows, image.cols, CV_32S, cv::Scalar(-1));
    for (std::size_t i = 0; i < yolo.detections.size(); ++i) {
        if (!yolo.detections[i].polygon.empty()) {
            // Same overlap rule as the deployed code: a later mask overwrites
            // an earlier mask because the server does not transmit confidence.
            cv::fillPoly(instance_map,
                         std::vector<std::vector<cv::Point>>{
                             yolo.detections[i].polygon},
                         cv::Scalar(static_cast<int>(i)));
        }
    }

    std::vector<ProjectedPoint> points;
    points.reserve(cloud->size());
    const double fx = intrinsics_(0, 0);
    const double fy = intrinsics_(1, 1);
    const double cx = intrinsics_(0, 2);
    const double cy = intrinsics_(1, 2);

    for (std::size_t raw_id = 0; raw_id < cloud->size(); ++raw_id) {
        const PointT& p = cloud->points[raw_id];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;
        if (std::abs(p.x) > crop_half_extent_ ||
            std::abs(p.y) > crop_half_extent_) continue;
        const double point_height = static_cast<double>(p.z) + lidar_height_;
        if (point_height < obstacle_height_threshold_) continue;

        const Eigen::Vector4d camera = camera_body_from_lidar_ *
            Eigen::Vector4d(p.x, p.y, p.z, 1.0);
        const double X = camera.x();
        const double Y = camera.y();
        const double Z = camera.z();
        if (X <= min_camera_depth_) continue;
        const int u = static_cast<int>(std::round(cx - fx * Y / X));
        const int v = static_cast<int>(std::round(cy - fy * Z / X));
        if (u < 0 || u >= image.cols || v < 0 || v >= image.rows) continue;

        ProjectedPoint ep;
        ep.lidar = p;
        ep.u = u;
        ep.v = v;
        ep.instance_id = instance_map.at<int>(v, u);
        if (ep.instance_id >= 0) {
            ep.class_id = yolo.detections.at(
                static_cast<std::size_t>(ep.instance_id)).class_id;
        }
        points.push_back(ep);
    }

    std::map<std::string, std::vector<uint8_t>> flags;
    const std::vector<std::string> principal_methods{
        "projection_only", "geometry_only", "largest_cluster", "proposed"};
    for (const auto& method : principal_methods)
        flags[method] = std::vector<uint8_t>(points.size(), 0);

    std::vector<std::size_t> all_associated;
    std::map<int, std::vector<std::size_t>> per_instance;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i].instance_id < 0) continue;
        all_associated.push_back(i);
        per_instance[points[i].instance_id].push_back(i);
        flags["projection_only"][i] = 1;
    }

    const auto geometry_clusters = clusterSubset(
        points, all_associated, default_radius_, default_min_cluster_size_,
        max_cluster_size_);
    markClusters(geometry_clusters, flags["geometry_only"], false);

    for (const auto& item : per_instance) {
        const auto clusters = clusterSubset(
            points, item.second, default_radius_, default_min_cluster_size_,
            max_cluster_size_);
        markClusters(clusters, flags["largest_cluster"], true);
        markClusters(clusters, flags["proposed"], false);
    }

    if (publish_clouds_) {
        publishMethod(points, flags.at("projection_only"), cloud_stamp,
                      projection_pub_);
        publishMethod(points, flags.at("geometry_only"), cloud_stamp,
                      geometry_pub_);
        publishMethod(points, flags.at("largest_cluster"), cloud_stamp,
                      largest_pub_);
        publishMethod(points, flags.at("proposed"), cloud_stamp, proposed_pub_);
    }
    ROS_INFO_STREAM("Processed " << key << ": " << points.size()
                    << " projectable points, " << all_associated.size()
                    << " YOLO-associated points.");
}

void ExperimentRecorder::publishMethod(
    const std::vector<ProjectedPoint>& points,
    const std::vector<uint8_t>& selected,
    const ros::Time& stamp,
    ros::Publisher& publisher) const {
    PointCloudT::Ptr lidar_cloud(new PointCloudT);
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (selected[i]) lidar_cloud->push_back(points[i].lidar);
    }
    PointCloudT::Ptr base_cloud(new PointCloudT);
    pcl::transformPointCloud(*lidar_cloud, *base_cloud, base_from_lidar_.matrix());
    sensor_msgs::PointCloud2 message;
    pcl::toROSMsg(*base_cloud, message);
    message.header.stamp = stamp;
    message.header.frame_id = "base_link";
    publisher.publish(message);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "rgb_lidar_experiment_recorder");
    ros::NodeHandle private_nh("~");
    ExperimentRecorder recorder(private_nh);
    ros::spin();
    return 0;
}
