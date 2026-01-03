// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"

#include <stdexcept>

namespace mlx::core {

namespace {

void throw_not_implemented(const std::string& op_name) {
  throw std::runtime_error(
      "[vulkan] Operation '" + op_name + "' not yet implemented on Vulkan backend");
}

}  // namespace

#define NOT_IMPLEMENTED(T) \
  void T::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw_not_implemented(#T); \
  }

// Unary operations
void Abs::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Abs");
}

void ArcCos::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcCos");
}

void ArcCosh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcCosh");
}

void ArcSin::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcSin");
}

void ArcSinh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcSinh");
}

void ArcTan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcTan");
}

void ArcTanh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcTanh");
}

void Ceil::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Ceil");
}

void Conjugate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Conjugate");
}

void Cos::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Cos");
}

void Cosh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Cosh");
}

void Erf::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Erf");
}

void ErfInv::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ErfInv");
}

void Exp::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Exp");
}

void Expm1::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Expm1");
}

void Floor::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Floor");
}

void Imag::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Imag");
}

void Log::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Log");
}

void Log2::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Log2");
}

void Log10::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Log10");
}

void Log1p::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Log1p");
}

void LogicalNot::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("LogicalNot");
}

void Negative::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Negative");
}

void Real::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Real");
}

void Round::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Round");
}

void Sigmoid::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Sigmoid");
}

void Sign::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Sign");
}

void Sin::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Sin");
}

void Sinh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Sinh");
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Sqrt");
}

void Square::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Square");
}

void Tan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Tan");
}

void Tanh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Tanh");
}

// Binary operations
void Add::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Add");
}

void ArcTan2::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArcTan2");
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("BitwiseBinary");
}

void Divide::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Divide");
}

void DivMod::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("DivMod");
}

void Equal::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Equal");
}

void Greater::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Greater");
}

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("GreaterEqual");
}

void Less::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Less");
}

void LessEqual::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("LessEqual");
}

void LogAddExp::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("LogAddExp");
}

void LogicalAnd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("LogicalAnd");
}

void LogicalOr::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("LogicalOr");
}

void Maximum::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Maximum");
}

void Minimum::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Minimum");
}

void Multiply::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Multiply");
}

void NaNEqual::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("NaNEqual");
}

void NotEqual::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("NotEqual");
}

void Power::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Power");
}

void Remainder::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Remainder");
}

void Subtract::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Subtract");
}

// Reduction operations
void Reduce::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Reduce");
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArgReduce");
}

void Scan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Scan");
}

// Matrix operations
void Matmul::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Matmul");
}

void AddMM::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("AddMM");
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("BlockMaskedMM");
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("GatherMM");
}

// Convolution
void Convolution::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Convolution");
}

// Attention
void ScaledDotProductAttention::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ScaledDotProductAttention");
}

// Normalization
void LayerNorm::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("LayerNorm");
}

void RMSNorm::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("RMSNorm");
}

void GroupNorm::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("GroupNorm");
}

void RoPE::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("RoPE");
}

// Data movement
void Copy::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Copy");
}

void Reshape::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Reshape");
}

void Transpose::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Transpose");
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Broadcast");
}

void BroadcastTo::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("BroadcastTo");
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Concatenate");
}

void Pad::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Pad");
}

void Split::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Split");
}

void Slice::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Slice");
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("SliceUpdate");
}

void Squeeze::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Squeeze");
}

void ExpandDims::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ExpandDims");
}

void Gather::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Gather");
}

void Scatter::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Scatter");
}

void Take::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Take");
}

void TakeAlongAxis::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("TakeAlongAxis");
}

// Type conversion
void AsType::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("AsType");
}

void View::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("View");
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("AsStrided");
}

// Creation
void Arange::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Arange");
}

void Full::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Full");
}

void NumberOfElements::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("NumberOfElements");
}

// Random
void RandomBits::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("RandomBits");
}

// Sorting
void Sort::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Sort");
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArgSort");
}

void Partition::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Partition");
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("ArgPartition");
}

// Linear algebra
void QRF::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("QRF");
}

void SVD::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("SVD");
}

void Inverse::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Inverse");
}

void Cholesky::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Cholesky");
}

void Eig::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Eig");
}

void Eigh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Eigh");
}

// Quantization
void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("QuantizedMatmul");
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("GatherQMM");
}

void AffineQuantize::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("AffineQuantize");
}

// Ternary
void Select::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Select");
}

void Where::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Where");
}

// FFT
void FFT::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("FFT");
}

// Softmax
void Softmax::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Softmax");
}

// Misc
void StopGradient::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("StopGradient");
}

void Compiled::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Compiled");
}

void Depends::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Depends");
}

void Load::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw_not_implemented("Load");
}

}  // namespace mlx::core
