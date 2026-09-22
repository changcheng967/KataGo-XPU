// MIGraphX backend for KataGo on Hygon DCU.
// Uses MIGraphX (the DCU's "TensorRT" — graph compiler with operator fusion)
// to run the whole neural network as a single compiled program, replacing the
// per-operation hipblas/MIOpen orchestration of the ROCm backend.
//
// Measured on Z200SM_80: ~64% faster inference than the ROCm backend for
// tf2-b10c384 (127.8 vs ~78 pos/s single GPU).
//
// Build: -DUSE_BACKEND=MIGRAPHX -DMIGRAPHX_ROOT=<dtk prefix>
// Usage: same as other backends, but -model must be an .onnx file (use
// `katago dumponnx` from an ONNX-backend build to produce it).

#include "../core/global.h"
#include "../core/logicerror.h"
#include "../core/logger.h"
#include "../core/makedirectory.h"
#include "../core/config_parser.h"
#include "../game/rules.h"
#include "../search/searchparams.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <migraphx/migraphx.h>

using namespace std;

#ifdef _WIN32
#define MGX_API __declspec(dllimport)
#else
#define MGX_API
#endif

// MIGraphX C API status check
#define MGX_CHECK(x, msg) do { \
    migraphx_status s_ = (x); \
    if(s_ != MIGRAPHX_STATUS_SUCCESS) { \
        throw StringError(string("MIGraphX error ") + to_string((int)s_) + ": " + msg); \
    } \
} while(0)

namespace NeuralNet {

struct MgxModel {
  migraphx_program_t program = nullptr;
  migraphx_target_t target = nullptr;
  migraphx_program_parameter_shapes_t inputShapes = nullptr;

  // Cached input names and element counts for fast batch setup
  struct InputInfo {
    string name;
    size_t elements;
    vector<size_t> lengths;
  };
  vector<InputInfo> inputs;

  int nnXLen = 19, nnYLen = 19;
  int maxBatchSize = 16;
  ~MgxModel();
};

MgxModel::~MgxModel() {
  if(program) migraphx_program_destroy(program);
  if(target) migraphx_target_destroy(target);
  if(inputShapes) migraphx_program_parameter_shapes_destroy(inputShapes);
}

struct MgxHandle {
  const MgxModel* model;
  int gpuIdx;
  ~MgxHandle() = default;
};

struct MgxContext {
  bool enabled = true;
  vector<MgxHandle*> handles;
  ~MgxContext();
};

MgxContext::~MgxContext() {
  for(auto* h : handles) delete h;
}

static MgxContext* mgxCtx = nullptr;

ComputeContext* createMigraphxContext(
  Logger* logger, const Config& cfg, bool shared
) {
  (void)cfg; (void)shared;
  if(mgxCtx == nullptr) mgxCtx = new MgxContext();
  return (ComputeContext*)mgxCtx;
}

void Migraphx_disable_fp16(ComputeContext* context) {
  ((MgxContext*)context)->enabled = false;
}

bool Migraphx_has_fp16(ComputeContext* context) {
  return ((MgxContext*)context)->enabled;
}

int Migraphx_get_gpus(ComputeContext* context) {
  return 1;  // Single GPU per context
}

ComputeHandle* Migraphx_create_compute_handle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdx,
  int serverIdx
) {
  (void)requireExactNNLen; (void)inputsUseNHWC; (void)serverIdx;

  auto* ctx = (MgxContext*)context;
  if(!ctx->enabled) throw StringError("MIGraphX backend not enabled");

  // Get model file path from LoadedModel
  const ModelDesc& desc = loadedModel->modelDesc;
  string modelFile = desc.fileName;
  if(modelFile.size() < 5 || modelFile.substr(modelFile.size()-5) != ".onnx") {
    throw StringError("MIGraphX backend requires an .onnx model file (use `katago dumponnx` to produce it): " + modelFile);
  }

  logger->write("MIGraphX backend: loading " + modelFile);

  auto* model = new MgxModel();
  model->maxBatchSize = maxBatchSize;
  model->nnXLen = desc.nnXLen;
  model->nnYLen = desc.nnYLen;

  // Parse ONNX
  migraphx_onnx_options_t onnxOpts;
  MGX_CHECK(migraphx_onnx_options_create(&onnxOpts), "init onnx options");
  MGX_CHECK(migraphx_parse_onnx(modelFile.c_str(), &onnxOpts, &model->program), "parse onnx");
  MGX_CHECK(migraphx_onnx_options_destroy(onnxOpts), "destroy onnx opts");

  // Compile for GPU
  logger->write("MIGraphX backend: compiling for GPU (this may take a while)...");
  auto t0 = chrono::steady_clock::now();
  MGX_CHECK(migraphx_target_create(&model->target, "gpu"), "create gpu target");
  migraphx_compile_options_t compileOpts;
  MGX_CHECK(migraphx_compile_options_create(&compileOpts), "init compile options");
  MGX_CHECK(migraphx_program_compile(model->program, model->target, compileOpts), "compile");
  MGX_CHECK(migraphx_compile_options_destroy(compileOpts), "destroy compile opts");
  auto elapsed = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
  logger->write("MIGraphX backend: compiled in " + Global::doubleToString(elapsed, 1) + "s");

  // Cache input parameter info
  MGX_CHECK(migraphx_program_get_parameter_shapes(model->program, &model->inputShapes), "get shapes");
  size_t nParams = 0;
  MGX_CHECK(migraphx_program_parameter_shapes_size(&nParams, model->inputShapes), "shapes size");
  for(size_t i = 0; i < nParams; i++) {
    const char* name = nullptr;
    MGX_CHECK(migraphx_program_parameter_shapes_get_name(&name, model->inputShapes, i), "get name");
    migraphx_shape_t shape = nullptr;
    MGX_CHECK(migraphx_program_parameter_shapes_get_shape(&shape, model->inputShapes, name), "get shape");
    size_t nelem = 0;
    MGX_CHECK(migraphx_shape_elements(&nelem, shape), "elements");
    size_t ndims = 0;
    MGX_CHECK(migraphx_shape_ndims(&ndims, shape), "ndims");
    vector<size_t> lens(ndims);
    for(size_t d = 0; d < ndims; d++) {
      const size_t* plen = nullptr;
      MGX_CHECK(migraphx_shape_lengths(&plen, shape), "lengths");
      lens[d] = plen[d];
    }
    MgxModel::InputInfo info;
    info.name = name;
    info.elements = nelem;
    info.lengths = lens;
    model->inputs.push_back(info);
    logger->write("MIGraphX backend: input " + string(name) + " shape has " + to_string(nelem) + " elements");
  }

  // Note: inputs are in model file order and should match KataGo's
  // InputSpatial, InputGlobal, InputMask ordering
  if(model->inputs.size() < 3) {
    throw StringError("MIGraphX backend: expected at least 3 inputs (spatial, global, mask)");
  }

  auto* handle = new MgxHandle();
  handle->model = model;
  handle->gpuIdx = gpuIdx;
  ctx->handles.push_back(handle);
  return (ComputeHandle*)handle;
}

void Migraphx_free_compute_handle(ComputeHandle* gpuHandle) {
  auto* h = (MgxHandle*)gpuHandle;
  auto* ctx = mgxCtx;
  if(ctx) {
    auto it = find(ctx->handles.begin(), ctx->handles.end(), h);
    if(it != ctx->handles.end()) ctx->handles.erase(it);
  }
  delete h;
}

// Main inference: run the compiled MIGraphX program on a batch of inputs.
// buffers[0]=InputSpatial [N,22,19,19], buffers[1]=InputGlobal [N,19,1,1],
// buffers[2]=InputMask [N,1,19,19] (all float, NHWC layout)
void Migraphx_handle_batch(
  ComputeHandle* gpuHandle,
  float* inputSpatial, float* inputGlobal, float* inputMask,
  float* policyOutput, float* valueOutput, float* scoreValueOutput,
  float* ownershipOutput, int batchLen,
  const NNResultBuf** inputBufs,
  std::mutex* mutexes
) {
  auto* h = (MgxHandle*)gpuHandle;
  const MgxModel* model = h->model;

  // Create parameters and bind input buffers
  migraphx_program_parameters_t params;
  MGX_CHECK(migraphx_program_parameters_create(&params), "create params");

  // Map KataGo buffers to model inputs by order (spatial, global, mask)
  float* buffers[3] = {inputSpatial, inputGlobal, inputMask};
  for(int i = 0; i < 3 && i < (int)model->inputs.size(); i++) {
    // Get the shape for this batch size
    migraphx_shape_t shape = nullptr;
    MGX_CHECK(migraphx_program_parameter_shapes_get_shape(
      &shape, model->inputShapes, model->inputs[i].name.c_str()), "get shape");
    MGX_CHECK(migraphx_program_parameters_add(
      params, model->inputs[i].name.c_str(), shape, buffers[i]), "add param");
  }

  // Run inference
  migraphx_arguments_t outputs = nullptr;
  MGX_CHECK(migraphx_program_run(&outputs, model->program, params), "run");

  // Extract outputs (policy, value, scoreValue, ownership)
  // Output order in the model: policy, value, scoreValue, ownership, misc
  auto copyOutput = [&](int idx, float* dst, size_t maxLen) {
    if(idx >= 4 || dst == nullptr) return;
    migraphx_argument_t arg = nullptr;
    MGX_CHECK(migraphx_arguments_get(&arg, outputs, idx), "get output");
    void* data = nullptr;
    MGX_CHECK(migraphx_argument_data(&data, arg), "get data");
    migraphx_shape_t shape = nullptr;
    MGX_CHECK(migraphx_argument_shape(&shape, arg), "get shape");
    size_t nelem = 0;
    MGX_CHECK(migraphx_shape_elements(&nelem, shape), "elements");
    size_t toCopy = nelem < maxLen ? nelem : maxLen;
    memcpy(dst, data, toCopy * sizeof(float));
  };

  size_t policyLen = (size_t)batchLen * 2 * 19 * 19;
  size_t valueLen = (size_t)batchLen * 3;
  size_t scoreLen = (size_t)batchLen * 6;
  size_t ownLen = (size_t)batchLen * 19 * 19;

  copyOutput(0, policyOutput, policyLen);
  copyOutput(1, valueOutput, valueLen);
  copyOutput(2, scoreValueOutput, scoreLen);
  copyOutput(3, ownershipOutput, ownLen);

  // Cleanup
  MGX_CHECK(migraphx_arguments_destroy(outputs), "destroy outputs");
  MGX_CHECK(migraphx_program_parameters_destroy(params), "destroy params");
}

int Migraphx_get_batch_size(ComputeHandle* gpuHandle) {
  return ((MgxHandle*)gpuHandle)->model->maxBatchSize;
}

bool Migraphx_supports_shortterm_error() { return false; }
bool Migraphx_using_fp16(ComputeHandle* gpuHandle) { return true; }
string Migraphx_backend_name() { return "MIGraphX"; }
void Migraphx_log_device_info(ComputeHandle* gpuHandle, Logger* logger) {
  logger->write("MIGraphX backend on Hygon DCU");
}

} // namespace NeuralNet
