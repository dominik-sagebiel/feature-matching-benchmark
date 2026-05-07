#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Helper to draw a small circle (filled) at (cx, cy) in RGB image
void draw_circle(unsigned char* img, int w, int h, int cx, int cy, int radius, 
                 unsigned char r, unsigned char g, unsigned char b) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            if (x >= 0 && x < w && y >= 0 && y < h && (dx*dx + dy*dy) <= radius*radius) {
                // RGB interleaved
                int idx = (y * w + x) * 3;
                img[idx]     = r;
                img[idx + 1] = g;
                img[idx + 2] = b;
            }
        }
    }
}

int main() {
    try {
        std::cout << "Loading model..." << std::endl;
        torch::jit::script::Module module = torch::jit::load("superpoint_batchsize1_traced.pt");
        module.eval();

        // Load image (grayscale)
        const char* image_path = "Resized_Image.png";
        int width, height, channels;
        unsigned char* img_gray = stbi_load(image_path, &width, &height, &channels, 1); // force grayscale
        if (!img_gray) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return -1;
        }
        std::cout << "Loaded " << width << "x" << height << " grayscale image" << std::endl;

        // Convert to tensor [1,1,H,W] float [0,1]
        torch::Tensor input = torch::zeros({1, 1, height, width});
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                input[0][0][y][x] = img_gray[y * width + x] / 255.0f;
            }
        }

        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        auto output = module.forward({input}).toTuple();
        auto end = std::chrono::high_resolution_clock::now();

        auto keypoints = output->elements()[0].toTensor();
        auto scores    = output->elements()[1].toTensor();
        auto descriptors = output->elements()[2].toTensor();

        std::cout << "Keypoints shape: " << keypoints.sizes() << std::endl;
        std::cout << "Scores shape: " << scores.sizes() << std::endl;
        std::cout << "Descriptors shape: " << descriptors.sizes() << std::endl;
        std::cout << "Inference time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << " ms" << std::endl;

        // --- Visualize keypoints ---
        // Create RGB image buffer (3 bytes per pixel)
        unsigned char* img_rgb = new unsigned char[width * height * 3];
        // Fill with grayscale values (to keep original appearance)
        for (int i = 0; i < width * height; ++i) {
            img_rgb[i*3]     = img_gray[i];
            img_rgb[i*3 + 1] = img_gray[i];
            img_rgb[i*3 + 2] = img_gray[i];
        }

        // Draw keypoints as red circles (radius 2)
        auto kp_acc = keypoints.accessor<float, 2>();
        for (int i = 0; i < keypoints.size(0); ++i) {
            int x = static_cast<int>(kp_acc[i][0]);
            int y = static_cast<int>(kp_acc[i][1]);
            draw_circle(img_rgb, width, height, x, y, 2, 255, 0, 0); // red
        }

        // Save result as PNG
        const char* output_path = "keypoints_output.png";
        if (stbi_write_png(output_path, width, height, 3, img_rgb, width * 3)) {
            std::cout << "Saved visualization to " << output_path << std::endl;
        } else {
            std::cerr << "Failed to write " << output_path << std::endl;
        }

        delete[] img_rgb;
        stbi_image_free(img_gray);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}