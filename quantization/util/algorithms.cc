#include "quantization/util/algorithms.h"
#include "quantization/util/helper.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace adaptive_system {

void apply_quantized_gradient_to_model(
    NamedGradientsAccordingColumn& named_gradients,
    tensorflow::Session* sess,
    Tuple& tuple,
    float const learning_rate_value) {
    google::protobuf::Map<std::string, GradientAccordingColumn>& map_gradient =
        *named_gradients.mutable_name_to_gradient();
    google::protobuf::Map<std::string, Names>& map_names =
        *tuple.mutable_map_names();
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
    std::vector<std::string> actions_to_do;
    actions_to_do.push_back(tuple.training_op_name());
    std::for_each(
        map_gradient.begin(), map_gradient.end(),
        [&feeds, &map_names, &tuple](
            google::protobuf::MapPair<std::string, GradientAccordingColumn>&
                pair) {
            std::string const& variable_name = pair.first;
            GradientAccordingColumn& grad = pair.second;
            auto iter_map_names = map_names.find(variable_name);
            if (iter_map_names == map_names.end()) {
                std::cout << "this is impossible Line " << __LINE__
                          << std::endl;
                std::terminate();
            } else {
                Names& names = iter_map_names->second;
                std::string grad_name = names.gradient_name();
                tensorflow::Tensor
                    feed_grad;  // nothing need to do to initialize feed
                                // tensor, dequantize function will do all
                                // stuff
                bool is_quantized = grad.is_quantized();
                if (is_quantized) {
                    dequantize_gradient_according_column(grad, feed_grad);
                } else {
                    feed_grad.FromProto(grad.tensor());
                }

                feeds.push_back(std::pair<std::string, tensorflow::Tensor>(
                    grad_name, feed_grad));
            }
        });
    tensorflow::Tensor learning_rate_tensor = tensorflow::Tensor(
        tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({}));
    float* learning_rate_tensor_ptr = learning_rate_tensor.flat<float>().data();
    *learning_rate_tensor_ptr = learning_rate_value;
    auto learning_rate_tensor_name = tuple.learning_rate_placeholder_name();
    feeds.push_back({learning_rate_tensor_name, learning_rate_tensor});
    tensorflow::Status status = sess->Run(feeds, {}, actions_to_do, nullptr);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    std::cout << "finished update!!!" << std::endl;
}

void apply_quantized_gradient_to_model(
    std::map<std::string, tensorflow::Tensor>& map_gradients,
    tensorflow::Session* sess,
    Tuple& tuple,
    float const learning_rate_value) {
    auto map_names = tuple.map_names();
    std::vector<std::pair<std::string, tensorflow::Tensor>> feed;
    for (auto iter = map_gradients.begin(); iter != map_gradients.end();
         iter++) {
        auto variable_name = iter->first;
        auto gradient_name =
            map_names.find(variable_name)->second.gradient_name();
        feed.push_back({gradient_name, iter->second});
    }
    tensorflow::Tensor learning_rate_tensor = tensorflow::Tensor(
        tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({}));
    float* learning_rate_tensor_ptr = learning_rate_tensor.flat<float>().data();
    *learning_rate_tensor_ptr = learning_rate_value;
    auto learning_rate_tensor_name = tuple.learning_rate_placeholder_name();
    feed.push_back({learning_rate_tensor_name, learning_rate_tensor});
    std::vector<std::string> actions_to_do;
    actions_to_do.push_back(tuple.training_op_name());
    tensorflow::Status status = sess->Run(feed, {}, actions_to_do, nullptr);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
}
void copy_variable_between_session(tensorflow::Session* session_from,
                                   tensorflow::Session* session_to,
                                   Tuple& tuple) {
    auto& map_names = tuple.map_names();
    std::vector<std::string> variable_names, assign_names,
        placeholder_assign_names;
    for (auto iter = map_names.begin(); iter != map_names.end(); iter++) {
        std::string variable_name = iter->first;
        std::string assign_name = iter->second.assign_name();
        std::string placeholder_assign_name =
            iter->second.placeholder_assign_name();
        variable_names.push_back(variable_name);
        assign_names.push_back(assign_name);
        placeholder_assign_names.push_back(placeholder_assign_name);
    }
    std::vector<tensorflow::Tensor> variable_results;
    tensorflow::Status status =
        session_from->Run({}, variable_names, {}, &variable_results);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
    int const size = placeholder_assign_names.size();
    for (int i = 0; i < size; i++) {
        auto pln = placeholder_assign_names[i];
        feeds.push_back({pln, variable_results[i]});
    }
    variable_results.clear();
    status = session_to->Run(feeds, assign_names, {}, &variable_results);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
}
namespace {
// all tensors in vec_gradient are in the same shape
void sum_two_tensors(tensorflow::Tensor& first_tensor,
                     tensorflow::Tensor& second_tensor,
                     tensorflow::Tensor& result) {
    int const tensor_length = first_tensor.NumElements();
    float* first_ptr = first_tensor.flat<float>().data();
    float* second_ptr = second_tensor.flat<float>().data();
    result = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
                                first_tensor.shape());
    float* result_ptr = result.flat<float>().data();
    for (int i = 0; i < tensor_length; i++) {
        result_ptr[i] = first_ptr[i] + second_ptr[i];
    }
}
void sum_gradient(std::vector<tensorflow::Tensor>& vec_gradient,
                  tensorflow::Tensor& result) {
    int const size = vec_gradient.size();
    if (size == 1) {
        result = vec_gradient[0];
        return;
    }
    if (size == 2) {
        auto& first_tensor = vec_gradient[0];
        auto& second_tensor = vec_gradient[1];
        sum_two_tensors(first_tensor, second_tensor, result);
        return;
    }
    tensorflow::Tensor first_tensor, second_tensor;
    std::vector<tensorflow::Tensor> first_half(vec_gradient.begin(),
                                               vec_gradient.begin() + size / 2),
        second_half(vec_gradient.begin() + size / 2, vec_gradient.end());
    std::thread thread_1(sum_gradient, std::ref(first_half),
                         std::ref(first_tensor));
    std::thread thread_2(sum_gradient, std::ref(second_half),
                         std::ref(second_tensor));
    thread_1.join();
    thread_2.join();
    sum_two_tensors(first_tensor, second_tensor, result);
}

void sum_gradient_by_multi_thread(
    std::vector<std::map<std::string, tensorflow::Tensor>>& vec_map_gradient,
    std::map<std::string, tensorflow::Tensor>& result) {
    std::map<std::string, std::vector<tensorflow::Tensor>> temp_map;
    for (auto iter = vec_map_gradient.begin(); iter != vec_map_gradient.end();
         iter++) {
        auto& map_gradient = *iter;
        for (auto iter_mg = map_gradient.begin(); iter_mg != map_gradient.end();
             iter_mg++) {
            temp_map[iter_mg->first].push_back(iter_mg->second);
        }
    }
    std::vector<std::thread> vec_thread;
    int const size = temp_map.size();
    for (auto iter = temp_map.begin(); iter != temp_map.end(); iter++) {
        std::string variable_name = iter->first;
        vec_thread.push_back(std::thread(sum_gradient,
                                         std::ref(temp_map[variable_name]),
                                         std::ref(result[variable_name])));
    }
    for (int i = 0; i < size; i++) {
        vec_thread[i].join();
    }
}
}  // namespace

std::map<int, std::pair<std::map<std::string, tensorflow::Tensor>, int>>&
iter_to_gradient() {
    static std::map<int,
                    std::pair<std::map<std::string, tensorflow::Tensor>, int>>
        iter2gradient;
    return iter2gradient;
}
void copy_gradient_between_session(int const last_iter,
                                   int const current_iter,
                                   int const total_worker,
                                   int const level,
                                   float const learning_rate_value,
                                   tensorflow::Session* session_from,
                                   tensorflow::Session* session_to,
                                   Tuple& tuple,
                                   int const threshold_to_quantize) {
    auto& iter2gradient = iter_to_gradient();
    std::vector<std::map<std::string, tensorflow::Tensor>> gradient_to_sum;
    for (int i = last_iter + 1; i <= current_iter; i++) {
        auto iter = iter2gradient.find(i);
        if (iter == iter2gradient.end()) {
            std::terminate();
        }
        gradient_to_sum.push_back(iter->second.first);
        iter->second.second++;
        if (iter->second.second == total_worker) {
            iter2gradient.erase(iter);
        }
    }
    // note that results is (variable_name, gradient_tensor)
    std::map<std::string, tensorflow::Tensor> results;
    sum_gradient_by_multi_thread(gradient_to_sum, results);
    NamedGradientsAccordingColumn named_gradients_send;
    quantize_gradients_according_column(results, &named_gradients_send, level,
                                        threshold_to_quantize);
    results.clear();
    dequantize_gradients_according_column(named_gradients_send, results);
    std::vector<std::pair<std::string, tensorflow::Tensor>> feed;
    for (auto iter = results.begin(); iter != results.end(); iter++) {
        std::string variable_name = iter->first;
        std::string gradient_name =
            tuple.map_names().find(variable_name)->second.gradient_name();
        feed.push_back({gradient_name, iter->second});
    }
    tensorflow::Tensor learning_rate_tensor = tensorflow::Tensor(
        tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({}));
    float* learning_rate_tensor_ptr = learning_rate_tensor.flat<float>().data();
    *learning_rate_tensor_ptr = learning_rate_value;
    auto learning_rate_tensor_name = tuple.learning_rate_placeholder_name();
    feed.push_back({learning_rate_tensor_name, learning_rate_tensor});
    std::vector<std::string> actions_to_do;
    actions_to_do.push_back(tuple.training_op_name());
    tensorflow::Status status =
        session_to->Run(feed, {}, actions_to_do, nullptr);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
}
// suitable for state like statistics of gradient
void moving_average(size_t length,
                    float const* previous,
                    float* current,
                    float const r) {
    for (size_t i = 0; i < length; i++) {
        current[i] = r * previous[i] + (1 - r) * current[i];
    }
}

float moving_average_v2(float const previous,
                        std::vector<float> const& losses,
                        std::vector<float>& new_losses,
                        float const r) {
    size_t size = losses.size();
    new_losses.resize(size);
    new_losses[0] = r * previous + (1 - r) * losses[0];
    for (size_t i = 1; i < size; i++) {
        new_losses[i] = r * new_losses[i - 1] + (1 - r) * losses[i];
    }
    return new_losses[size - 1];
}

void moving_average_then_minus_average(std::vector<float> const& losses,
                                       std::vector<float>& new_losses,
                                       float const r) {
    size_t size = losses.size();
    new_losses.resize(size);
    new_losses[0] = losses[0];
    for (int i = 1; i < size; i++) {
        new_losses[i] = r * new_losses[i - 1] + (1 - r) * losses[i];
    }
    float sum = std::accumulate(new_losses.begin(), new_losses.end(), 0.0f);
    float average = sum / size;
    for (int i = 0; i < size; i++) {
        new_losses[i] -= average;
    }
}

void moving_average_not_minus_average(std::vector<float> const& losses,
                                      std::vector<float>& new_losses,
                                      float const r) {
    size_t size = losses.size();
    new_losses.resize(size);
    new_losses[0] = losses[0];
    for (int i = 1; i < size; i++) {
        new_losses[i] = r * new_losses[i - 1] + (1 - r) * losses[i];
    }
}

float moving_average_from_last_loss(float const last_loss,
                                    std::vector<float> const& losses,
                                    std::vector<float>& new_losses,
                                    float const r) {
    int size = losses.size();
    new_losses.resize(size);
    new_losses[0] = r * last_loss + (1 - r) * losses[0];
    for (int i = 1; i < size; i++) {
        new_losses[i] = r * new_losses[i - 1] + (1 - r) * losses[i];
    }
    return new_losses[size - 1];
}

float minus_average_then_moving_average(std::vector<float> const& losses,
                                        std::vector<float>& new_losses,
                                        float const r) {
    size_t size = losses.size();
    new_losses.resize(size);
    float sum = std::accumulate(losses.begin(), losses.end(), 0.0f);
    float average = sum / size;
    std::vector<float> temp;
    for (float f : losses) {
        temp.push_back(f - average);
    }
    new_losses[0] = temp[0];
    for (size_t i = 1; i < size; i++) {
        new_losses[i] = r * new_losses[i - 1] + (1 - r) * temp[i];
    }
    return new_losses[size - 1];
}

void standard_times(std::vector<float>& times) {
    size_t size = times.size();
    float base = times[0];
    for (int i = 0; i < size; i++) {
        times[i] -= base;
    }
}

tensorflow::Tensor get_feed_tensor_from_action(int action_order  // begin from 0
) {
    const size_t total_actions = 3;
    tensorflow::Tensor ret(tensorflow::DataType::DT_FLOAT,
                           tensorflow::TensorShape({total_actions}));
    float* ret_ptr = ret.flat<float>().data();
    std::fill(ret_ptr, ret_ptr + total_actions, 0.0f);
    ret_ptr[action_order] = 1.0;
    return ret;
}

namespace {
bool greater_compare_pair(std::pair<std::string, int> const& a,
                          std::pair<std::string, int> const& b) {
    return b.second < a.second;
}

}  // namespace

float get_slope(std::vector<float> const& times,
                std::vector<float> const& move_average_losses) {
    using namespace Eigen;
    int const size = times.size();
    std::cout << "time is ::" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << times[i] << "  ";
    }
    std::cout << std::endl;

    std::cout << "average is ::" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << move_average_losses[i] << "  ";
    }
    std::cout << std::endl;
    MatrixXf A = MatrixXf::Random(size, 2);
    VectorXf b = VectorXf::Random(size);
    for (int i = 0; i < size; i++) {
        A(i, 0) = times[i];
        A(i, 1) = 1.0f;
        b(i) = move_average_losses[i];
    }
    // std::cout << A << std::endl << b << std::endl;
    auto qr = A.fullPivHouseholderQr();
    auto w = qr.solve(b);
    std::cout << "slope is " << w << std::endl;
    return w(0);
}

void average_gradients(
    int const number_workers,
    std::map<std::string, tensorflow::Tensor>& name2gradient) {
    auto begin = name2gradient.begin();
    auto end = name2gradient.end();
    for (auto iter = begin; iter != end; iter++) {
        tensorflow::Tensor& tensor = iter->second;
        float* tensor_ptr = tensor.flat<float>().data();
        size_t size = tensor.NumElements();
        std::for_each(tensor_ptr, tensor_ptr + size,
                      [number_workers](float& current) {
                          current = current / number_workers;
                      });
    }
}

int get_real_level(int const order, int const level) {
    int temp = 0;
    const int min_level = 6;
    const int max_level = 10;
    if (order == 0) {
        temp = level - 1;
        return temp < min_level ? min_level : temp;
    } else if (order == 1) {
        return level;
    } else if (order == 2) {
        temp = level + 1;
        return temp > max_level ? max_level : temp;
    }
    PRINT_ERROR_MESSAGE("order must between 0 and 2");
    std::terminate();
}

int get_real_level_6_8_10(int order) {
    switch (order) {
        case 0:
            return 6;
        case 1:
            return 8;
        case 2:
            return 10;
    }
    PRINT_ERROR_MESSAGE("order should be in the range of 0 to 2");
    std::terminate();
}

namespace {
void sum_tensor_vector(std::vector<tensorflow::Tensor> const& vec_tensor,
                       tensorflow::Tensor& out_tensor) {
    auto shape = vec_tensor[0].shape();
    tensorflow::Tensor return_tensor(tensorflow::DataType::DT_FLOAT, shape);
    size_t size = return_tensor.NumElements();
    // std::cout << "size is " << size << std::endl;
    float* return_tensor_ptr = return_tensor.flat<float>().data();
    std::fill(return_tensor_ptr, return_tensor_ptr + size, 0.0f);
    for (int i = 0; i < vec_tensor.size(); i++) {
        auto& tensor = vec_tensor[i];
        float const* tensor_ptr = tensor.flat<float>().data();
        for (int j = 0; j < size; j++) {
            return_tensor_ptr[j] += tensor_ptr[j];
        }
    }
    out_tensor = std::move(return_tensor);
}
}  // namespace

void aggregate_gradients(
    std::vector<std::map<std::string, tensorflow::Tensor>>& vector_of_map,
    std::map<std::string, tensorflow::Tensor>& return_result) {
    std::map<std::string, std::vector<tensorflow::Tensor>> map_tensor_vector;
    PRINT_INFO;
    for (auto iter = vector_of_map.begin(); iter != vector_of_map.end();
         iter++) {
        std::map<std::string, tensorflow::Tensor>& map_current = *iter;
        for (auto iter_name_tensor = map_current.begin();
             iter_name_tensor != map_current.end(); iter_name_tensor++) {
            std::string var_name = iter_name_tensor->first;
            tensorflow::Tensor& gradient = iter_name_tensor->second;
            map_tensor_vector[var_name].push_back(gradient);
        }
    }
    PRINT_INFO;
    std::vector<std::thread> vector_threads;
    std::vector<std::pair<std::string, tensorflow::Tensor>> vector_name_tensor;
    vector_name_tensor.resize(map_tensor_vector.size());
    int index = 0;
    for (auto iter = map_tensor_vector.begin(); iter != map_tensor_vector.end();
         iter++) {
        std::string var_name = iter->first;
        auto& vector_tensor = iter->second;
        vector_name_tensor[index].first = var_name;
        auto& ref_tensor = vector_name_tensor[index++].second;
        vector_threads.push_back(std::thread(
            sum_tensor_vector, std::ref(vector_tensor), std::ref(ref_tensor)));
    }
    PRINT_INFO;
    for (auto iter = vector_threads.begin(); iter != vector_threads.end();
         iter++) {
        iter->join();
    }
    PRINT_INFO;
    return_result.clear();
    for (auto iter = vector_name_tensor.begin();
         iter != vector_name_tensor.end(); iter++) {
        return_result[iter->first] = iter->second;
    }
    PRINT_INFO;
}

tensorflow::Tensor get_float_tensor_from_vector(const std::vector<float>& vec) {
    int size = vec.size();
    tensorflow::Tensor ret(tensorflow::DataType::DT_FLOAT,
                           tensorflow::TensorShape({size}));
    float* ptr_ret = ret.flat<float>().data();
    for (int i = 0; i < size; i++) {
        ptr_ret[i] = vec[i];
    }
    return ret;
}

float get_slope_according_loss(const std::vector<float>& loss_vec) {
    using namespace Eigen;
    int const size = loss_vec.size();
    std::cout << "loss is ::" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << loss_vec[i] << "  ";
    }
    std::cout << std::endl;

    MatrixXf A = MatrixXf::Random(size, 2);
    VectorXf b = VectorXf::Random(size);
    for (int i = 0; i < size; i++) {
        A(i, 0) = i;
        A(i, 1) = 1.0f;
        b(i) = loss_vec[i];
    }
    // std::cout << A << std::endl << b << std::endl;
    auto qr = A.fullPivHouseholderQr();
    auto w = qr.solve(b);
    std::cout << "slope is " << w << std::endl;
    return w(0);
}
}  // namespace adaptive_system
