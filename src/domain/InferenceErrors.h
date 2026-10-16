#pragma once

#include <stdexcept>
#include <string>
#include <utility>

class InvalidInferenceRequest : public std::invalid_argument
{
public:
    InvalidInferenceRequest(std::string code, std::string message)
        : std::invalid_argument(std::move(message)), code_(std::move(code))
    {
    }

    const std::string& code() const noexcept { return code_; }

private:
    std::string code_;
};

namespace triton_backend
{

class BackendNotReady : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace triton_backend
