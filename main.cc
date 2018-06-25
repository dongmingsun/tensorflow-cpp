/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// #include <opencv2/core/core.hpp>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc.hpp>

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::uint8;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string &file_name, std::vector<string> *result,
                      size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env *env, const string &filename,
                             Tensor *output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string &file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor> *out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  // auto float_caster =
  //     Cast(root.WithOpName("float_caster"), image_reader,
  //     tensorflow::DT_FLOAT);

  auto uint8_caster =
      Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  // auto resized = ResizeBilinear(
  //     root, dims_expander,
  //     Const(root.WithOpName("size"), {input_height, input_width}));

  // Subtract the mean and divide by the scale.
  // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander,
  // {input_mean}),
  //     {input_std});

  // cast to int
  // auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div,
  // tensorflow::DT_UINT8);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
  // declare the graph
  std::cout << "[Status] start loading model" << std::endl;
  tensorflow::GraphDef graph_def;
  // the actual load process: ReadBinaryProto
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  // reset
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  // feed the model to session
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor> &outputs, int how_many_labels,
                    Tensor *indices, Tensor *scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor> &outputs,
                      const string &labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor> &outputs, int expected,
                     bool *is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

Status ReframeBoxMasksToImageMasks(tensorflow::Tensor &intput_masks,
                                   tensorflow::Tensor &input_boxes,
                                   int num_detections,
                                   std::vector<Tensor> *out_tensors, int height,
                                   int width) {
  /*
   *intput_masks: [1,100,15,15]
   *input_boxes: [1,100,4]
   */
  string output_name = "output";
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;

  // Prepare masks for CropAndResize
  // Shape: [1,6,15,15]
  auto masks_reduced =
      Slice(root, intput_masks, {0, 0, 0, 0}, {1, num_detections, 15, 15});
  // Shape: [6,15,15,1]
  auto masks_reshaped =
      Reshape(root, masks_reduced, {num_detections, 15, 15, 1});
  // Prepare masks for CropAndResize done

  // Prepare boxes for CropAndResize
  // Shape: [1,6,2]
  auto min_corner = Slice(root, input_boxes, tensorflow::Input{0, 0, 0},
                          tensorflow::Input{1, num_detections, 2});
  auto max_corner = Slice(root, input_boxes, tensorflow::Input{0, 0, 2},
                          tensorflow::Input{1, num_detections, 2});
  // Shape: [6,1,2]
  auto min_corner_reshaped = Reshape(root, min_corner, {num_detections, 1, 2});
  auto max_corner_reshaped = Reshape(root, max_corner, {num_detections, 1, 2});

  // Prepare unit boxes
  tensorflow::TensorShape unit_boxes_shape({num_detections, 2});
  tensorflow::Tensor unit_boxes_zeros(tensorflow::DT_FLOAT, unit_boxes_shape);
  tensorflow::Tensor unit_boxes_ones(tensorflow::DT_FLOAT, unit_boxes_shape);
  for (int row = 0; row < num_detections; ++row) {
    for (int col = 0; col < 2; ++col) {
      unit_boxes_zeros.tensor<float, 2>()(row, col) = 0;
    }
  }
  for (int row = 0; row < num_detections; ++row) {
    for (int col = 0; col < 2; ++col) {
      unit_boxes_ones.tensor<float, 2>()(row, col) = 1;
    }
  }
  auto unit_boxes = Concat(root, {unit_boxes_zeros, unit_boxes_ones}, 1);
  // Unit boxes shape: [6,2,2]
  auto unit_boxes_reshaped = Reshape(root, unit_boxes, {-1, 2, 2});
  auto transformed_boxes =
      Div(root, Sub(root, unit_boxes_reshaped, min_corner_reshaped),
          Sub(root, max_corner_reshaped, min_corner_reshaped));
  auto transformed_boxes_reshaped = Reshape(root, transformed_boxes, {-1, 4});
  // Prepare boxes for CropAndResize done

  // Prepare box_ind for CropAndResize
  tensorflow::TensorShape box_ind_shape({num_detections});
  tensorflow::Tensor box_ind(tensorflow::DT_INT32, box_ind_shape);
  auto box_ind_vec = box_ind.vec<int32>();
  for (int i = 0; i < num_detections; ++i) {
    box_ind_vec(i) = i;
  }

  // Produce the masks reframed to the image we need
  auto masks_reframed =
      CropAndResize(root, masks_reshaped, transformed_boxes_reshaped, box_ind,
                    {height, width});
  auto masks_squeezed = Squeeze(root.WithOpName(output_name), masks_reframed);

  // This runs the GraphDef network definition that we've just
  // constructed, and returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {"output"}, {}, out_tensors));
  return Status::OK();
}

Status SaveAsJPG(const tensorflow::Tensor &input, string file_name) {
  /*
   *input: [1,height,width,3]
   */
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;
  // Shape: [height, width, 3]
  // @input_squeezed shape: [height, width, 3]
  auto input_squeezed = Squeeze(root, input);
  // @input_encoded: 0-D. JPEG-encoded image.
  auto input_encoded = EncodeJpeg(root, input_squeezed);
  auto created_operation =
      WriteFile(root.WithOpName("output/image"), file_name, input_encoded);

  // Run the GraphDef network
  // std::vector<Tensor> *out_tensors;
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {}, {"output/image"}, {}));
  return Status::OK();
}

Status SaveMasksAsJPG(const tensorflow::Tensor &input, int mask_number,
                      string file_name, int height, int width) {
  /*
   *input: [6,width,height]
   */
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;

  // @first_mask shape: [1, height, width, 3]
  auto first_mask =
      Reshape(root, Slice(root, input, {mask_number, 0, 0}, {1, height, width}),
              {height, width, 1});
  float threshold = 0.5;
  auto first_mask_cast = Cast(root, GreaterEqual(root, first_mask, {threshold}),
                              tensorflow::DT_UINT8);
  tensorflow::uint8 scale = 200;
  auto first_mask_scaled = Multiply(root, first_mask_cast, {scale});
  // @input_encoded: 0-D. JPEG-encoded image.
  auto input_encoded = EncodeJpeg(root, first_mask_scaled);
  auto created_operation =
      WriteFile(root.WithOpName("output/image"), file_name, input_encoded);

  // Run the GraphDef network
  // std::vector<Tensor> *out_tensors;
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {}, {"output/image"}, {}));
  return Status::OK();
}
int main(int argc, char *argv[]) {
  string image = "../data/image1.jpg";
  string graph = "../model/mask_rcnn_inception_v2_coco_2018_01_28/"
                 "frozen_inference_graph.pb";
  // TODO load labels
  string labels = "../data/imagenet_slim_labels.txt";
  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  // TODO check layer name
  string input_layer = "image_tensor:0";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool self_test = false;
  string root_dir = "";

  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels, "name of file containing labels"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // NOTE Load the model
  std::unique_ptr<tensorflow::Session> session; // init session
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  std::cout << "[Status] load model sucess" << std::endl;

  // TODO check load image code
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  // @resized_tensor: the tensor storing the image
  const Tensor &resized_tensor = resized_tensors[0];
  auto resized_tensor_height = resized_tensor.shape().dim_sizes()[1];
  auto resized_tensor_width = resized_tensor.shape().dim_sizes()[2];
  auto resized_tensor_channels = resized_tensor.shape().dim_sizes()[3];

  std::cout << "height:\t\t" << resized_tensor_height << "\nwidth:\t\t"
            << resized_tensor_width << "\nchannels:\t"
            << resized_tensor_channels << std::endl;

  // Run the Mask R-CNN model
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {"num_detections:0", "detection_boxes:0",
                                    "detection_scores:0", "detection_classes:0",
                                    "detection_masks:0"},
                                   {}, &outputs); // original:{output_layer}
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  int num_detections = (int)(outputs[0].scalar<float>()(0));

  // TODO Check reframe masks
  std::vector<Tensor> image_masks;
  Status reframe_box_masks_status = ReframeBoxMasksToImageMasks(
      outputs[4], outputs[1], num_detections, &image_masks,
      resized_tensor_height, resized_tensor_width);
  if (!reframe_box_masks_status.ok()) {
    LOG(ERROR) << reframe_box_masks_status;
    return -1;
  }

  std::cout << "\n==============================\n"
            << "detection_masks:0"
            << "\n==============================\n"
            << image_masks[0].DebugString() << std::endl;
  // auto image_masks_tensor = image_masks[0].tensor<float, 3>();
  for (int i = 0; i < num_detections; ++i) {
    std::cout << "Saving mask_" << i + 1 << ".jpg"
              << "\n";
    const string file_name = "mask_" + std::to_string(i + 1) + ".jpg";
    auto save_mask_flag =
        SaveMasksAsJPG(image_masks[0], i, file_name, resized_tensor_height,
                       resized_tensor_width);
    if (!save_mask_flag.ok()) {
      LOG(ERROR) << save_mask_flag;
      return -1;
    }
  }
  std::cout << std::endl;

  return 0;
}

// NOTE convert output[i] to tensors
// auto output_detection_classes = outputs[3].tensor<float, 2>();
// std::cout << "detection classes" << std::endl;
// for (int i = 0; i < 5; ++i) {
//   std::cout << output_detection_classes(0, i) << "\n";
// }
//
// auto output_detection_boxes = outputs[1].tensor<float, 3>();
// std::cout << "detection boxes" << std::endl;
// std::cout << std::fixed;
// for (int i = 0; i < 10; ++i) {
//   for (int j = 0; j < 4; ++j)
//     std::cout << std::setprecision(4) <<
//     output_detection_boxes(0, i, j)
//               << "\t";
//   std::cout << std::endl;
// }

// NOTE Self test code
// This is for automated testing to make sure we get the expected result with
// the default settings. We know that label 653 (military uniform) should be
// the top label for the Admiral Hopper image.
// if (self_test) {
//   bool expected_matches;
//   Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
//   if (!check_status.ok()) {
//     LOG(ERROR) << "Running check failed: " << check_status;
//     return -1;
//   }
//   if (!expected_matches) {
//     LOG(ERROR) << "Self-test failed!";
//     return -1;
//   }
// }
