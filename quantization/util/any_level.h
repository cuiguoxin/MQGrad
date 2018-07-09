#ifndef ADAPTIVE_SYSTEM_ANY_LEVEL
#define ADAPTIVE_SYSTEM_ANY_LEVEL
#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include "proto/rpc_service.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace adaptive_system {

	void quantize_gradient(uint32_t const level, tensorflow::Tensor const& tensor, Gradient& gradient);

	void dequantize_gradient(Gradient const & gradient, tensorflow::Tensor & tensor);

	void quantize_gradients(std::map<std::string, tensorflow::Tensor>& map_gradient,
		NamedGradients* named_gradients, int level);

	void dequantize_gradients(
		NamedGradients& named_gradients,
		std::map<std::string, tensorflow::Tensor>& map_gradient);
	
	void quantize_gradient_according_column(uint32_t const level, tensorflow::Tensor const & tensor,
		GradientAccordingColumn& gradient);

	void dequantize_gradient_according_column(GradientAccordingColumn const & gradient,
		tensorflow::Tensor& tensor);

	void quantize_gradients_according_column(std::map<std::string, tensorflow::Tensor>& map_gradient,
		NamedGradientsAccordingColumn* named_gradients, int level, int threshold);

	void dequantize_gradients_according_column(NamedGradientsAccordingColumn& named_gradients,
		std::map<std::string, tensorflow::Tensor>& map_gradient);

}
#endif // !ADAPTIVE_SYSTEM_ANY_LEVEL
