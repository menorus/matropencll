#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <string>
#include <chrono>

const char* kernelSource = R"(
    __kernel void matrixMultiply(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int M,
        const int N,
        const int K)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
)";

bool checkOpenCLError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error in " << operation << ": " << err << std::endl;
        return false;
    }
    return true;
}

void multiplyMatricesOpenCL(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K) {
    std::vector<cl::Platform> platforms;
    if (!checkOpenCLError(cl::Platform::get(&platforms), "getting platforms")) {
        exit(1);
    }

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        exit(1);
    }

    cl::Platform platform;
    bool found = false;
    for (auto& p : platforms) {
        std::string vendor = p.getInfo<CL_PLATFORM_VENDOR>();
        if (vendor.find("Advanced Micro Devices") != std::string::npos) {
            platform = p;
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "AMD platform not found, using first available" << std::endl;
        platform = platforms[0];
    }

    std::vector<cl::Device> devices;
    if (!checkOpenCLError(platform.getDevices(CL_DEVICE_TYPE_GPU, &devices), "getting devices")) {
        exit(1);
    }
    if (devices.empty()) {
        std::cerr << "No GPU devices found" << std::endl;
        exit(1);
    }

    cl::Device device = devices[0];
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * A.size(), (void*)A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * B.size(), (void*)B.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * C.size(), nullptr);

    cl::Program::Sources sources;
    sources.push_back({ kernelSource, strlen(kernelSource) });
    cl::Program program(context, sources);
    cl_int buildErr = program.build({ device });
    if (!checkOpenCLError(buildErr, "building program")) {
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    cl::Kernel kernel(program, "matrixMultiply");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, M);
    kernel.setArg(4, N);
    kernel.setArg(5, K);

    cl::NDRange globalSize(M, N);
    cl::NDRange localSize(8, 8);

    auto start = std::chrono::high_resolution_clock::now();

    if (!checkOpenCLError(queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize),
        "enqueueing kernel")) {
        exit(1);
    }
    if (!checkOpenCLError(queue.enqueueReadBuffer(bufferC, CL_TRUE, 0,
        sizeof(float) * C.size(), C.data()),
        "reading buffer")) {
        exit(1);
    }
    queue.finish();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "OpenCL matrix multiplication took: "
        << duration.count() / 1000.0 << " milliseconds" << std::endl;
}

void multiplyMatricesCPU(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU matrix multiplication took: "
        << duration.count() / 1000.0 << " milliseconds" << std::endl;
}

int main() {
    const int M = 512, K = 512, N = 512;
    std::vector<float> A(M * K), B(K * N), C_OpenCL(M * N), C_CPU(M * N);

    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>(i + 1);

    multiplyMatricesCPU(A, B, C_CPU, M, N, K);
    multiplyMatricesOpenCL(A, B, C_OpenCL, M, N, K);

    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(C_OpenCL[i] - C_CPU[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;

    std::cout << "First few elements of C (CPU):" << std::endl;
    for (int i = 0; i < (M < 5 ? M : 5); i++) {
        for (int j = 0; j < (N < 5 ? N : 5); j++) {
            std::cout << C_CPU[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}