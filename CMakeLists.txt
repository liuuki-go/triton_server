cmake_minimum_required(VERSION 3.20)
project(TritonBack VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(TRITON_ROOT "/opt/triton/2.71.0" CACHE PATH "Validated Triton runtime")
set(TRITON_MODEL_REPOSITORY "${PROJECT_SOURCE_DIR}/models" CACHE PATH "Model repository for integration tests")

find_path(TRITON_INCLUDE_DIR triton/core/tritonserver.h PATHS "${TRITON_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TRITON_SERVER_LIBRARY tritonserver PATHS "${TRITON_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CPP_HTTPLIB REQUIRED IMPORTED_TARGET cpp-httplib)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(domain INTERFACE)
target_include_directories(domain INTERFACE "${PROJECT_SOURCE_DIR}/src")

add_library(service_contract INTERFACE)
target_link_libraries(service_contract INTERFACE domain nlohmann_json::nlohmann_json)

add_library(image_processing STATIC src/inference/RawImagePreprocessing.cpp src/inference/ImagePipeline.cpp)
target_include_directories(image_processing PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_processing PUBLIC domain ${OpenCV_LIBS})

add_library(backend_config STATIC src/configuration/BackendConfig.cpp)
target_link_libraries(backend_config PUBLIC service_contract yaml-cpp)

add_library(backend_api STATIC src/api/BackendHttpServer.cpp src/api/MetadataCodec.cpp src/api/BoundedHttpTaskQueue.cpp)
target_link_libraries(backend_api PUBLIC service_contract backend_config PkgConfig::CPP_HTTPLIB Threads::Threads)

add_library(triton_service STATIC src/services/impl/BackendService.cpp)
target_include_directories(triton_service PUBLIC "${TRITON_INCLUDE_DIR}")
target_link_libraries(triton_service PUBLIC service_contract image_processing Threads::Threads "${TRITON_SERVER_LIBRARY}")

add_executable(triton_backend src/app/main.cpp)
target_link_libraries(triton_backend PRIVATE backend_api triton_service)
target_compile_definitions(triton_backend PRIVATE DEFAULT_CONFIG_PATH="${PROJECT_SOURCE_DIR}/config/server.yaml")
set_target_properties(triton_backend PROPERTIES BUILD_RPATH "${TRITON_ROOT}/lib" INSTALL_RPATH "${TRITON_ROOT}/lib")

foreach(target image_processing backend_config backend_api triton_service triton_backend)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()

include(CTest)
if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)

    function(add_unit_test target source library)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE ${library} GTest::gtest_main)
        gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit TIMEOUT 30)
    endfunction()

    add_unit_test(test_raw_image tests/unit/testRawImage.cpp image_processing)
    add_unit_test(test_image_pipeline tests/unit/testImagePipeline.cpp image_processing)
    add_unit_test(test_backend_config tests/unit/testBackendConfig.cpp backend_config)
    add_unit_test(test_backend_http tests/unit/testBackendHttpServer.cpp backend_api)
    target_link_libraries(test_backend_http PRIVATE image_processing)
    add_unit_test(test_backend_service_config tests/unit/testBackendServiceConfig.cpp triton_service)

    add_executable(test_backend_service tests/integration/testBackendService.cpp)
    target_link_libraries(test_backend_service PRIVATE triton_service GTest::gtest_main)
    target_compile_definitions(test_backend_service PRIVATE
        TEST_CONFIG_ROOT="${PROJECT_SOURCE_DIR}/config"
        TEST_FIXTURE_ROOT="${PROJECT_SOURCE_DIR}/tests/fixtures"
        TEST_MODEL_REPOSITORY="${TRITON_MODEL_REPOSITORY}")
    gtest_discover_tests(test_backend_service TEST_FILTER "BackendRuntimeTest.*"
        DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu TIMEOUT 90)
endif()

install(TARGETS triton_backend RUNTIME DESTINATION bin)
install(DIRECTORY config models DESTINATION .)
