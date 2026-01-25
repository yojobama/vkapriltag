#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>

extern "C" {
#include "vulkan_apriltag/vulkan_apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagStandard41h12.h"
#include "common/getopt.h"
}

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_string(getopt, 'i', "image", "", "Input image file (if empty, use camera)");
    getopt_add_int(getopt, 'c', "camera", "0", "camera ID (used if no image specified)");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, 'g', "gpu", 1, "Use GPU acceleration");
    getopt_add_bool(getopt, 'p', "performance", 0, "Show performance comparison");
    getopt_add_string(getopt, 'o', "output", "", "Output image file (optional)");

    if (!getopt_parse(getopt, argc, argv, 1) ||
            getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        printf("Examples:\n");
        printf("  %s -i apriltag.png                    # Process image file\n", argv[0]);
        printf("  %s -i apriltag.png -o result.png      # Process and save result\n", argv[0]);
        printf("  %s -c 0                               # Use camera (default)\n", argv[0]);
        printf("  %s -i apriltag.png -g 0 -p            # CPU-only with performance stats\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

    bool use_gpu = getopt_get_bool(getopt, "gpu");
    bool show_performance = getopt_get_bool(getopt, "performance");
    const char* image_path = getopt_get_string(getopt, "image");
    const char* output_path = getopt_get_string(getopt, "output");
    bool use_image_file = strlen(image_path) > 0;

    // Check Vulkan support
    if (use_gpu && !vulkan_apriltag_is_supported()) {
        cout << "Vulkan not supported, falling back to CPU implementation" << endl;
        use_gpu = false;
    }

    // Initialize Vulkan context if using GPU
    vulkan_apriltag_context_t* vulkan_ctx = nullptr;
    vulkan_apriltag_detector_t* gpu_detector = nullptr;
    
    if (use_gpu) {
        cout << "Initializing Vulkan AprilTag detector..." << endl;
        vulkan_ctx = vulkan_apriltag_context_create(getopt_get_bool(getopt, "debug"));
        if (!vulkan_ctx) {
            cout << "Failed to create Vulkan context, falling back to CPU" << endl;
            use_gpu = false;
        } else {
            vulkan_apriltag_print_device_info(vulkan_ctx);
            
            // Create GPU detector with reasonable maximum dimensions
            gpu_detector = vulkan_apriltag_detector_create(vulkan_ctx, 4096, 4096);
            if (!gpu_detector) {
                cout << "Failed to create GPU detector, falling back to CPU" << endl;
                use_gpu = false;
            }
        }
    }

    TickMeter meter;
    meter.start();

    // Initialize input source
    VideoCapture cap;
    Mat static_image;
    bool use_camera = false;

    if (use_image_file) {
        cout << "Loading image: " << image_path << endl;
        static_image = imread(image_path, IMREAD_COLOR);
        if (static_image.empty()) {
            cerr << "Error: Could not load image from " << image_path << endl;
            return -1;
        }
        cout << "Image loaded: " << static_image.cols << "x" << static_image.rows << endl;
    } else {
        cout << "Enabling video capture" << endl;
        cap.open(getopt_get_int(getopt, "camera"));
        if (!cap.isOpened()) {
            cerr << "Couldn't open video capture device" << endl;
            return -1;
        }
        use_camera = true;
    }

    // Initialize CPU detector for comparison or fallback
    apriltag_detector_t *cpu_detector = apriltag_detector_create();
    
    // Initialize tag family
    apriltag_family_t *tf = nullptr;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11")) {
        tf = tag36h11_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }

    // Add family to detectors
    apriltag_detector_add_family(cpu_detector, tf);
    if (gpu_detector) {
        vulkan_apriltag_detector_add_family(gpu_detector, tf);
    }

    // Configure detectors
    cpu_detector->quad_decimate = getopt_get_double(getopt, "decimate");
    cpu_detector->quad_sigma = getopt_get_double(getopt, "blur");
    cpu_detector->debug = getopt_get_bool(getopt, "debug");
    
    if (gpu_detector) {
        vulkan_apriltag_detector_set_quad_decimate(gpu_detector, getopt_get_double(getopt, "decimate"));
        vulkan_apriltag_detector_set_quad_sigma(gpu_detector, getopt_get_double(getopt, "blur"));
        vulkan_apriltag_detector_set_debug(gpu_detector, getopt_get_bool(getopt, "debug"));
    }

    meter.stop();
    cout << "Detector " << famname << " initialized in "
        << std::fixed << std::setprecision(3) << meter.getTimeSec() << " seconds" << endl;
    
    if (use_gpu) {
        cout << "Using GPU acceleration" << endl;
    } else {
        cout << "Using CPU implementation" << endl;
    }

    if (use_camera) {
#if CV_MAJOR_VERSION > 3
        cout << "Camera: " << cap.get(CAP_PROP_FRAME_WIDTH ) << "x" <<
                        cap.get(CAP_PROP_FRAME_HEIGHT ) << " @" <<
                        cap.get(CAP_PROP_FPS) << "FPS" << endl;
#else
        cout << "Camera: " << cap.get(CV_CAP_PROP_FRAME_WIDTH ) << "x" <<
                        cap.get(CV_CAP_PROP_FRAME_HEIGHT ) << " @" <<
                        cap.get(CV_CAP_PROP_FPS) << "FPS" << endl;
#endif
    }

    Mat frame, gray;
    double total_cpu_time = 0.0;
    double total_gpu_time = 0.0;
    int frame_count = 0;
    bool processing_complete = false;
    
    while (!processing_complete) {
        // Get frame
        if (use_camera) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            frame = static_image.clone();
            processing_complete = true; // Process image once and exit
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data
        image_u8_t im = {gray.cols, gray.rows, gray.cols, gray.data};

        zarray_t *detections = nullptr;
        
        auto start_time = chrono::high_resolution_clock::now();
        
        if (use_gpu && gpu_detector) {
            // GPU detection
            detections = vulkan_apriltag_detector_detect(gpu_detector, &im);
            
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            total_gpu_time += duration.count() / 1000.0; // Convert to milliseconds
            
            if (show_performance || !use_camera) {
                uint64_t gpu_time_ns = vulkan_apriltag_detector_get_gpu_time_ns(gpu_detector);
                cout << "GPU detection time: " << duration.count() / 1000.0 << " ms "
                     << "(GPU compute: " << gpu_time_ns / 1000000.0 << " ms)" << endl;
            }
        } else {
            // CPU detection
            detections = apriltag_detector_detect(cpu_detector, &im);
            
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            total_cpu_time += duration.count() / 1000.0;
            
            if (show_performance || !use_camera) {
                cout << "CPU detection time: " << duration.count() / 1000.0 << " ms" << endl;
            }
        }
        
        frame_count++;

        // Print detection results
        cout << "Found " << zarray_size(detections) << " tags:" << endl;
        
        // Draw detection outlines and print results
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            
            // Print detection info
            cout << "  Tag " << det->id << ": center=(" << std::fixed << std::setprecision(2) 
                 << det->c[0] << "," << det->c[1] << "), corners=[";
            for (int j = 0; j < 4; j++) {
                cout << "(" << det->p[j][0] << "," << det->p[j][1] << ")";
                if (j < 3) cout << ",";
            }
            cout << "], hamming=" << det->hamming << ", margin=" << det->decision_margin << endl;
            
            // Draw detection outline
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(0, 0xff, 0), 2);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 0, 0xff), 2);
            line(frame, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0xff, 0, 0), 2);
            line(frame, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0xff, 0, 0), 2);

            // Draw tag ID
            stringstream ss;
            ss << det->id;
            String text = ss.str();
            int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
            double fontscale = 1.0;
            int baseline;
            Size textsize = getTextSize(text, fontface, fontscale, 2, &baseline);
            putText(frame, text, Point(det->c[0]-textsize.width/2,
                                       det->c[1]+textsize.height/2),
                    fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
        }
        
        // Show performance stats on frame for camera mode
        if (show_performance && use_camera) {
            string perf_text;
            if (use_gpu) {
                perf_text = "GPU: " + to_string(total_gpu_time / frame_count) + " ms/frame";
            } else {
                perf_text = "CPU: " + to_string(total_cpu_time / frame_count) + " ms/frame";
            }
            putText(frame, perf_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, 
                   Scalar(255, 255, 255), 2);
        }
        
        apriltag_detections_destroy(detections);

        // Save output image if specified
        if (!use_camera && strlen(output_path) > 0) {
            if (imwrite(output_path, frame)) {
                cout << "Output image saved to: " << output_path << endl;
            } else {
                cerr << "Error: Failed to save output image to " << output_path << endl;
            }
        }

        // Display image
        string window_title = use_camera ? "Vulkan AprilTag Demo (Live)" : "Vulkan AprilTag Demo";
        imshow(window_title, frame);
        
        if (use_camera) {
            char key = waitKey(1);
            if (key >= 0) {
                if (key == 'g' && vulkan_ctx) {
                    // Toggle between GPU and CPU
                    use_gpu = !use_gpu;
                    cout << "Switched to " << (use_gpu ? "GPU" : "CPU") << " implementation" << endl;
                } else {
                    break;
                }
            }
        } else {
            // For static images, wait for key press to exit
            cout << "Press any key to exit..." << endl;
            waitKey(0);
        }
    }

    // Print final performance comparison
    if (show_performance && frame_count > 0) {
        cout << "\nPerformance Summary:" << endl;
        if (total_gpu_time > 0) {
            cout << "  Average GPU time: " << total_gpu_time / frame_count << " ms/frame" << endl;
        }
        if (total_cpu_time > 0) {
            cout << "  Average CPU time: " << total_cpu_time / frame_count << " ms/frame" << endl;
        }
        if (total_gpu_time > 0 && total_cpu_time > 0) {
            double speedup = total_cpu_time / total_gpu_time;
            cout << "  GPU speedup: " << speedup << "x" << endl;
        }
    }

    // Cleanup
    if (gpu_detector) {
        vulkan_apriltag_detector_destroy(gpu_detector);
    }
    if (vulkan_ctx) {
        vulkan_apriltag_context_destroy(vulkan_ctx);
    }
    
    apriltag_detector_destroy(cpu_detector);

    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    }

    getopt_destroy(getopt);

    return 0;
}