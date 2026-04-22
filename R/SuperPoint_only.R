# ============================================
# SuperPoint(rpautrat) in R (Torch) 
# ============================================
# Expected folder layout:
# myproject/
#   ├── Images/          # place input images here
#   ├── Python/
#   ├── R/               # this script lives here
#   ├── Repos/
#   │   ├── SuperPoint/
#   │   │   └── weights/ # contains superpoint_v6_from_tf.pth
#   │   └── SuperGluePretrainedNetwork/
#   └── Results/
#       └── R/           # outputs will be saved here

# ===== USER INPUT: specify which image to process =====
image_filename <- "HEstain.png"   # <-- CHANGE THIS to your image file name
# The file must be located in the 'Images/' folder.

# =======================================================

# Install required packages if missing
required_pkgs <- c("torch", "magick", "here")
for (pkg in required_pkgs) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ---- Get script directory robustly ----
get_script_dir <- function() {
  if (interactive()) {
    if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
      return(dirname(rstudioapi::getActiveDocumentContext()$path))
    }
    return(getwd())
  }
  if (!is.null(sys.frame(1)$ofile)) {
    return(dirname(sys.frame(1)$ofile))
  }
  return(getwd())
}

script_dir <- get_script_dir()
# Since script is in myproject/R/, go one level up
root_dir <- dirname(script_dir)   # = feature-matching-benchmark
repos_dir   <- file.path(root_dir, "Repos")
superpoint_dir <- file.path(repos_dir, "SuperPoint")
weights_file <- file.path(superpoint_dir, "weights", "superpoint_v6_from_tf.pth")

images_dir <- file.path(root_dir, "Images")
results_dir <- file.path(root_dir, "Results", "R", "rpautratSuperpoint")

# Create Results/R if it doesn't exist
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# ---- Check prerequisites ----
if (!dir.exists(superpoint_dir)) {
  stop("SuperPoint repo not found at: ", superpoint_dir,
       "\nPlease clone it into Repos/ first.")
}
if (!file.exists(weights_file)) {
  stop("Weight file not found: ", weights_file,
       "\nMake sure superpoint_v6_from_tf.pth is inside SuperPoint/weights/")
}
if (!dir.exists(images_dir)) {
  stop("Images folder not found: ", images_dir,
       "\nPlease create an 'Images' folder at the project root.")
}

# Build full path to the chosen image
image_path <- file.path(images_dir, image_filename)
if (!file.exists(image_path)) {
  stop("Image file not found: ", image_path,
       "\nPlease ensure the file exists in the 'Images' folder.")
}
cat("Using image:", image_path, "\n")

# ============================================
# Model definition
# ============================================

VGGBlock <- torch::nn_module(
  "VGGBlock",
  initialize = function(c_in, c_out, kernel_size, relu = TRUE) {
    padding <- floor((kernel_size - 1) / 2)
    self$conv <- nn_conv2d(c_in, c_out, 
                           kernel_size = kernel_size, 
                           stride = 1, 
                           padding = padding)
    if(relu) {
      self$activation <- nn_relu(inplace = TRUE)
    } else {
      self$activation <- nn_identity()
    }
    self$bn <- nn_batch_norm2d(c_out, eps = 0.001)
  },
  forward = function(x) {
    x <- self$conv(x)
    x <- self$activation(x)
    x <- self$bn(x)
    return(x)
  }
)

BackboneBlock <- torch::nn_module(
  "BackboneBlock",
  initialize = function(c_in, c_out, add_pooling = TRUE) {
    self$`0` <- VGGBlock(c_in, c_out, 3)
    self$`1` <- VGGBlock(c_out, c_out, 3)
    if(add_pooling) {
      self$`2` <- nn_max_pool2d(kernel_size = 2, stride = 2)
    }
  },
  forward = function(x) {
    x <- self$`0`(x)
    x <- self$`1`(x)
    if(!is.null(self$`2`)) {
      x <- self$`2`(x)
    }
    return(x)
  }
)

SuperPoint <- torch::nn_module(
  "SuperPoint",
  initialize = function(descriptor_dim = 256,
                        channels = c(64, 64, 128, 128),
                        nms_radius = 3,
                        detection_threshold = 0.0001,
                        max_num_keypoints = 5000,
                        remove_borders = 4) {
    self$descriptor_dim <- descriptor_dim
    self$nms_radius <- nms_radius
    self$detection_threshold <- detection_threshold
    self$max_num_keypoints <- max_num_keypoints
    self$remove_borders <- remove_borders
    self$channels <- channels
    self$stride <- 2 ^ (length(channels) - 1)
    
    backbone_blocks <- list()
    c_in <- 1
    for(i in 1:length(channels)) {
      c_out <- channels[i]
      add_pool <- (i < length(channels))
      backbone_blocks <- append(backbone_blocks, 
                                BackboneBlock(c_in, c_out, add_pool))
      c_in <- c_out
    }
    self$backbone <- nn_module_list(backbone_blocks)
    
    self$detector <- nn_module_list(list(
      VGGBlock(channels[length(channels)], 256, 3),
      VGGBlock(256, 65, 1, relu = FALSE)
    ))
    
    self$descriptor <- nn_module_list(list(
      VGGBlock(channels[length(channels)], 256, 3),
      VGGBlock(256, descriptor_dim, 1, relu = FALSE)
    ))
  },
  forward = function(image) {
    if(image$size(2) == 3) {
      scale <- torch_tensor(c(0.299, 0.587, 0.114), 
                            device = image$device())$view(c(1, 3, 1, 1))
      image <- torch_sum(image * scale, dim = 2, keepdim = TRUE)
    }
    
    features <- image
    for(i in 1:length(self$backbone)) {
      features <- self$backbone[[i]](features)
    }
    
    descriptors_dense <- self$descriptor[[1]](features)
    descriptors_dense <- self$descriptor[[2]](descriptors_dense)
    descriptors_dense <- nnf_normalize(descriptors_dense, p = 2, dim = 2)
    
    scores_logits <- self$detector[[1]](features)
    scores_logits <- self$detector[[2]](scores_logits)
    scores <- nnf_softmax(scores_logits, dim = 2)
    
    scores <- scores[, 1:(self$stride^2), , , drop = FALSE]
    
    batch_size <- as.integer(scores$size(1))
    h <- as.integer(scores$size(3))
    w <- as.integer(scores$size(4))
    orig_h <- as.integer(image$size(3))
    orig_w <- as.integer(image$size(4))
    
    scores <- scores$permute(c(1, 3, 4, 2))
    scores <- scores$reshape(c(batch_size, h, w, self$stride, self$stride))
    scores <- scores$permute(c(1, 2, 4, 3, 5))
    scores_full <- scores$reshape(c(batch_size, h * self$stride, w * self$stride))
    scores_full <- scores_full[, 1:orig_h, 1:orig_w]
    
    if(self$nms_radius > 0) {
      kernel_size <- self$nms_radius * 2 + 1
      padding <- self$nms_radius
      scores_4d <- scores_full$unsqueeze(1)
      max_pooled <- nnf_max_pool2d(scores_4d, kernel_size = kernel_size, stride = 1, padding = padding)
      max_pooled <- max_pooled$squeeze(1)
      max_mask <- scores_full == max_pooled
      scores_full <- torch_where(max_mask, scores_full, torch_zeros_like(scores_full))
    }
    
    keypoints_list <- list()
    scores_list <- list()
    descriptors_list <- list()
    
    for(b in 1:batch_size) {
      score_map <- as_array(scores_full[b, , ])
      
      if(self$remove_borders > 0) {
        pad <- self$remove_borders
        if(pad <= nrow(score_map)) {
          score_map[1:pad, ] <- 0
          score_map[(nrow(score_map) - pad + 1):nrow(score_map), ] <- 0
        }
        if(pad <= ncol(score_map)) {
          score_map[, 1:pad] <- 0
          score_map[, (ncol(score_map) - pad + 1):ncol(score_map)] <- 0
        }
      }
      
      high_points <- which(score_map > self$detection_threshold, arr.ind = TRUE)
      if(nrow(high_points) == 0) {
        keypoints_list[[b]] <- matrix(0, nrow=0, ncol=2)
        scores_list[[b]] <- numeric(0)
        descriptors_list[[b]] <- matrix(0, nrow=0, ncol=self$descriptor_dim)
        next
      }
      
      keypoints <- high_points[, c(2, 1), drop = FALSE]
      kp_scores <- score_map[high_points]
      
      if(nrow(keypoints) > 0) {
        max_kp <- self$max_num_keypoints
        if(!is.null(max_kp) && is.numeric(max_kp) && length(max_kp) == 1 && 
           !is.na(max_kp) && max_kp > 0 && max_kp < nrow(keypoints)) {
          order_idx <- order(kp_scores, decreasing = TRUE)[1:max_kp]
          order_idx <- order_idx[!is.na(order_idx)]
          if(length(order_idx) > 0) {
            keypoints <- keypoints[order_idx, , drop = FALSE]
            kp_scores <- kp_scores[order_idx]
          }
        }
      }
      
      desc_array <- as_array(descriptors_dense[b, , , ])
      kp_x <- floor(keypoints[, 1] / self$stride) + 1
      kp_y <- floor(keypoints[, 2] / self$stride) + 1
      kp_x <- pmax(1, pmin(w, kp_x))
      kp_y <- pmax(1, pmin(h, kp_y))
      
      descriptors_out <- matrix(0, nrow = nrow(keypoints), ncol = self$descriptor_dim)
      for(j in 1:nrow(keypoints)) {
        descriptors_out[j, ] <- desc_array[, kp_y[j], kp_x[j]]
      }
      desc_norm <- sqrt(rowSums(descriptors_out^2))
      desc_norm[desc_norm == 0] <- 1
      descriptors_out <- descriptors_out / desc_norm
      
      keypoints_list[[b]] <- keypoints
      scores_list[[b]] <- kp_scores
      descriptors_list[[b]] <- descriptors_out
    }
    
    return(list(
      keypoints = keypoints_list,
      keypoint_scores = scores_list,
      descriptors = descriptors_list
    ))
  }
)

# ============================================
# Load model
# ============================================
cat("Loading model from:", weights_file, "\n")
model <- SuperPoint()
state_dict <- torch::load_state_dict(weights_file)
model$load_state_dict(state_dict)
model$eval()

# ============================================
# Process image with memory safety
# ============================================
process_image_safe <- function(image_path, model, max_pixels = 8000000) {
  start_time <- Sys.time()
  img <- image_read(image_path)
  img_gray <- image_convert(img, colorspace = "gray")
  img_info <- image_info(img_gray)
  orig_w <- img_info$width
  orig_h <- img_info$height
  total_pixels <- orig_w * orig_h
  stride <- 8
  
  cat("Original size:", orig_w, "x", orig_h, "=", total_pixels, "pixels\n")
  
  if(total_pixels > max_pixels) {
    scale <- sqrt(max_pixels / total_pixels)
    new_w <- floor(orig_w * scale)
    new_h <- floor(orig_h * scale)
    new_w <- floor(new_w / stride) * stride
    new_h <- floor(new_h / stride) * stride
    cat("Downsampling to:", new_w, "x", new_h, "\n")
    img_resized <- image_resize(img_gray, paste0(new_w, "x", new_h))
  } else {
    new_w <- orig_w
    new_h <- orig_h
    img_resized <- img_gray
  }
  
  final_w <- floor(new_w / stride) * stride
  final_h <- floor(new_h / stride) * stride
  if(final_w != new_w || final_h != new_h) {
    cat("Trimming to:", final_w, "x", final_h, "\n")
    img_processed <- image_resize(img_resized, geometry = paste0(final_w, "x", final_h))
  } else {
    img_processed <- img_resized
  }
  preprocess_time <- Sys.time() - start_time
  
  start_time <- Sys.time()
  img_array <- as.numeric(img_processed[[1]]) / 255
  img_to_array_time <- Sys.time() - start_time
  start_time <- Sys.time()
  img_tensor <- torch_tensor(img_array)$view(c(1, 1, nrow(img_array), ncol(img_array)))
  array_to_torchtensor_time <- Sys.time() - start_time
  
  rm(img, img_gray, img_resized, img_array)
  gc()
  
  cat("Running inference...\n")
  start_time <- Sys.time()
  result <- model(img_tensor)
  inference_time <- Sys.time() - start_time
  cat("\n✅ Detected", nrow(result$keypoints[[1]]), "keypoints\n")
  
  total_time <- preprocess_time + img_to_array_time + array_to_torchtensor_time + inference_time
  
  # Create timestamp
  timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  
  # Create filename with timestamp
  txt_filename <- file.path(results_dir, paste0("superpoint_timings_", timestamp, ".txt"))
  
  # Get original image size (before resizing)
  original_h <- orig_h
  original_w <- orig_w
  
  # Get resized image size (after processing)
  resized_h <- new_h
  resized_w <- new_w
  
  # Save timings in same format as Python
  writeLines(c(
    "SuperPoint Performance Timings (R)",
    "========================================",
    "",
    paste("Image:", image_path),
    paste("Image original shape:", original_h, ",", original_w),
    "",
    paste("Image shape (resized):", resized_h, ",", resized_w),
    "",
    paste("Keypoints detected:", nrow(keypoints)),
    "",
    "Timings:",
    paste("  Preprocessing time:", round(as.numeric(preprocess_time), 4), "seconds"),
    paste("  Image to array time:", round(as.numeric(img_to_array_time), 4), "seconds"),
    paste("  Array to torch tensor time:", round(as.numeric(array_to_torchtensor_time), 4), "seconds"),
    paste("  Inference time:", round(as.numeric(inference_time), 4), "seconds"),
    paste("  Total time:", round(as.numeric(total_time), 4), "seconds")
  ), txt_filename)
  
  cat("✅ Timings saved to:", txt_filename, "\n")
  
  return(list(
    keypoints = result$keypoints[[1]],
    scores = result$keypoint_scores[[1]],
    descriptors = result$descriptors[[1]],
    image = img_processed,
    width = final_w,
    height = final_h
  ))
}

# ============================================
# Run processing
# ============================================
max_pixels <- 4000000  # adjust if needed (8 megapixels)
result <- process_image_safe(image_path, model, max_pixels)

keypoints <- result$keypoints
scores <- result$scores
img_processed <- result$image

# ============================================
# Save results to Results/R/ with timestamp
# ============================================
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")  

# Visualization
result_filename <- file.path(results_dir, paste0("superpoint_", timestamp, ".png"))
img_h <- result$height
img_w <- result$width

png(result_filename, width = img_w, height = img_h)
par(mar = c(0,0,0,0))
plot(1, type = "n", xlim = c(0, img_w), ylim = c(img_h, 0), axes = FALSE, xlab = "", ylab = "")
rasterImage(as.raster(img_processed), 0, img_h, img_w, 0)
points(keypoints[, 1], keypoints[, 2], col = "red", pch = 16, cex = 0.6)
dev.off()
cat("✅ Saved visualization:", result_filename, "\n")

# Statistics
stats_filename <- file.path(results_dir, paste0("superpoint_stats_", timestamp, ".txt"))
writeLines(
  c(
    paste("SuperPoint Results -", Sys.time()),
    paste("Image:", image_path),
    paste("Processed size:", result$width, "x", result$height),
    paste("Keypoints detected:", nrow(keypoints)),
    paste("Score range:", round(min(scores), 5), "to", round(max(scores), 5)),
    paste("Mean score:", round(mean(scores), 5)),
    paste("Detection threshold:", model$detection_threshold),
    paste("NMS radius:", model$nms_radius),
    paste("Max pixels limit:", max_pixels)
  ),
  stats_filename
)
cat("✅ Saved stats:", stats_filename, "\n")