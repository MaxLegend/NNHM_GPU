#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <cstdint>
#include <fstream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using uint = unsigned int;

// Настройки нейросети
const int TILE_SIZE = 64;
const int INPUT_SIZE = TILE_SIZE * TILE_SIZE;
const int HIDDEN_SIZE = 512;
const float GLOBAL_LEARNING_RATE = 0.05f; // Немного увеличим для ускорения
bool g_debugLoggingEnabled = true;

struct LayerConfig {
    uint32_t inputDim;
    uint32_t outputDim;
    float learningRate;
    uint32_t padding;
};

// [Global::ThrowIfFailed]
inline void ThrowIfFailed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        std::cerr << "[Global::ThrowIfFailed] Error: " << msg << " (HR: 0x" << std::hex << hr << ")" << std::endl;
        throw std::exception(msg);
    }
}

// --- HLSL Код ---
const char* hlslCode = R"(
cbuffer Config : register(b0) {
    uint inputDim;
    uint outputDim;
    float learningRate;
    uint padding;
};

StructuredBuffer<float> FwdInput : register(t0);
StructuredBuffer<float> FwdWeights : register(t1);
StructuredBuffer<float> FwdBias : register(t2);
RWStructuredBuffer<float> FwdOutput : register(u0);

[numthreads(256, 1, 1)]
void CSForward(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= outputDim) return;
    float sum = 0;
    for (uint i = 0; i < inputDim; ++i) {
        sum += FwdInput[i] * FwdWeights[i * outputDim + dtid.x];
    }
    FwdOutput[dtid.x] = 1.0f / (1.0f + exp(-(sum + FwdBias[dtid.x])));
}

StructuredBuffer<float> BkOutPred : register(t0);
StructuredBuffer<float> BkOutTarget : register(t1);
RWStructuredBuffer<float> BkOutGrad : register(u0);

[numthreads(256, 1, 1)]
void CSBackwardOutput(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= outputDim) return;
    float y = BkOutPred[dtid.x];
    BkOutGrad[dtid.x] = (y - BkOutTarget[dtid.x]) * y * (1.0f - y);
}

StructuredBuffer<float> BkHidNextGrad : register(t0);
StructuredBuffer<float> BkHidNextWeights : register(t1);
StructuredBuffer<float> BkHidCurrentAct : register(t2);
RWStructuredBuffer<float> BkHidCurrentGrad : register(u0);

[numthreads(256, 1, 1)]
void CSBackwardHidden(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= inputDim) return;
    float sum = 0;
    for (uint i = 0; i < outputDim; ++i) {
        sum += BkHidNextGrad[i] * BkHidNextWeights[dtid.x * outputDim + i];
    }
    float act = BkHidCurrentAct[dtid.x];
    BkHidCurrentGrad[dtid.x] = sum * act * (1.0f - act);
}

StructuredBuffer<float> UpdInput : register(t0);
StructuredBuffer<float> UpdGrad : register(t1);
RWStructuredBuffer<float> UpdWeights : register(u0);

[numthreads(256, 1, 1)]
void CSUpdateWeights(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= inputDim * outputDim) return;
    uint i = dtid.x / outputDim;
    uint j = dtid.x % outputDim;
    UpdWeights[dtid.x] -= learningRate * (UpdInput[i] * UpdGrad[j]);
}

StructuredBuffer<float> UpdBiasGrad : register(t0);
RWStructuredBuffer<float> UpdBias : register(u0);

[numthreads(256, 1, 1)]
void CSUpdateBias(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= outputDim) return;
    UpdBias[dtid.x] -= learningRate * UpdBiasGrad[dtid.x];
}
)";

// --- DX12 Core ---
struct DX12Context {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceVal = 0;
    HANDLE fenceEvent;

    void Init() {
        ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)), "Device creation failed");
        D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
        device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (g_debugLoggingEnabled) std::cout << "[DX12Context::Init] Success: GPU Online. Prepare dataset..." << std::endl;
    }

    void Sync() {
        list->Close();
        ID3D12CommandList* pp[] = { list.Get() };
        queue->ExecuteCommandLists(1, pp);
        UINT64 v = ++fenceVal;
        queue->Signal(fence.Get(), v);
        if (fence->GetCompletedValue() < v) {
            fence->SetEventOnCompletion(v, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
    }

    ComPtr<ID3D12Resource> CreateBuf(UINT64 sz, D3D12_HEAP_TYPE h, D3D12_RESOURCE_STATES s) {
        ComPtr<ID3D12Resource> r;
        D3D12_HEAP_PROPERTIES hp = { h, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
        D3D12_RESOURCE_DESC rd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, sz, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, (h == D3D12_HEAP_TYPE_DEFAULT) ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE };
        ThrowIfFailed(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, s, nullptr, IID_PPV_ARGS(&r)), "Buffer creation failed");
        return r;
    }
    ComPtr<ID3D12Resource> CreateReadbackBuf(UINT64 sz) {
        ComPtr<ID3D12Resource> r;
        D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
        D3D12_RESOURCE_DESC rd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, sz, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
        // Состояние обязано быть COPY_DEST для Readback буферов
        ThrowIfFailed(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r)), "CreateReadbackBuf failed");
        return r;
    }
};

struct Pipeline {
    ComPtr<ID3D12RootSignature> sig;
    ComPtr<ID3D12PipelineState> pso;
};

Pipeline CreatePSO(DX12Context* ctx, const char* entry, int numSRV, int numUAV) {
    Pipeline p;
    ComPtr<ID3DBlob> b, e;
    if (FAILED(D3DCompile(hlslCode, strlen(hlslCode), nullptr, nullptr, nullptr, entry, "cs_5_0", 0, 0, &b, &e))) {
        std::cerr << "[Global::CreatePSO] Shader Error: " << (char*)e->GetBufferPointer() << std::endl;
        throw std::exception("Shader Compile Error");
    }

    std::vector<D3D12_ROOT_PARAMETER> params;

    // Параметр 0: Root Constants (4 uint32_t = 16 байт)
    D3D12_ROOT_PARAMETER rc = {};
    rc.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rc.Constants.ShaderRegister = 0;
    rc.Constants.Num32BitValues = sizeof(LayerConfig) / 4;
    params.push_back(rc);

    for (int i = 0; i < numSRV; ++i) {
        D3D12_ROOT_PARAMETER s = {};
        s.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        s.Descriptor.ShaderRegister = i;
        params.push_back(s);
    }
    for (int i = 0; i < numUAV; ++i) {
        D3D12_ROOT_PARAMETER u = {};
        u.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        u.Descriptor.ShaderRegister = i;
        params.push_back(u);
    }

    D3D12_ROOT_SIGNATURE_DESC rsd = { (uint)params.size(), params.data(), 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
    ComPtr<ID3DBlob> s, er;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &s, &er), "SerializeRootSig");
    ThrowIfFailed(ctx->device->CreateRootSignature(0, s->GetBufferPointer(), s->GetBufferSize(), IID_PPV_ARGS(&p.sig)), "CreateRootSig");

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = { p.sig.Get(), { b->GetBufferPointer(), b->GetBufferSize() } };
    ThrowIfFailed(ctx->device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&p.pso)), "CreatePSO");

    if (g_debugLoggingEnabled) std::cout << "[Global::CreatePSO] Success: " << entry << std::endl;
    return p;
}

// [Global::SaveGPUPreview]
void SaveGPUPreview(DX12Context& ctx, ID3D12Resource* gpuRes, uint32_t step) {
    auto rb = ctx.CreateBuf(INPUT_SIZE * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ctx.list->CopyBufferRegion(rb.Get(), 0, gpuRes, 0, INPUT_SIZE * 4);
    ctx.Sync();
    std::vector<float> pix(INPUT_SIZE);
    void* p; rb->Map(0, nullptr, &p); memcpy(pix.data(), p, INPUT_SIZE * 4); rb->Unmap(0, nullptr);
    std::vector<unsigned char> img(INPUT_SIZE);
    for(int i=0; i<INPUT_SIZE; ++i) img[i] = (unsigned char)(std::clamp(pix[i], 0.0f, 1.0f) * 255.0f);
    stbi_write_png(("step_" + std::to_string(step) + ".png").c_str(), TILE_SIZE, TILE_SIZE, 1, img.data(), TILE_SIZE);
    std::cout << "[Global::SaveGPUPreview] Success: step_" << step << ".png saved." << std::endl;
}
void InitWeights(DX12Context& ctx, ComPtr<ID3D12Resource> dest, ComPtr<ID3D12Resource> upBuffer, uint32_t count) {
    if (dest == nullptr || upBuffer == nullptr) {
        std::cerr << "[Main::InitWeights] Error: Resource is null!" << std::endl;
        return;
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<float> v(count);
    for (auto& f : v) f = dist(gen);

    void* p = nullptr;
    HRESULT hr = upBuffer->Map(0, nullptr, &p);
    if (FAILED(hr) || p == nullptr) {
        std::cerr << "[Main::InitWeights] Error: Map failed (0x" << std::hex << hr << ")" << std::endl;
        return;
    }

    memcpy(p, v.data(), count * 4);
    upBuffer->Unmap(0, nullptr);

    ctx.list->CopyBufferRegion(dest.Get(), 0, upBuffer.Get(), 0, count * 4);
    ctx.Sync();

    if (g_debugLoggingEnabled) {
        std::cout << "[Main::InitWeights] Success: Buffer initialized (" << count << " floats)." << std::endl;
    }
}
void LoadGPUWeights(DX12Context& ctx, ComPtr<ID3D12Resource> w1, ComPtr<ID3D12Resource> b1, ComPtr<ID3D12Resource> w2, ComPtr<ID3D12Resource> b2, ComPtr<ID3D12Resource> up, const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) return;

    auto LoadToGPU = [&](ComPtr<ID3D12Resource> res, uint32_t size) {
        std::vector<float> data(size);
        f.read((char*)data.data(), size * 4);
        void* p;
        up->Map(0, nullptr, &p);
        memcpy(p, data.data(), size * 4);
        up->Unmap(0, nullptr);
        ctx.list->CopyBufferRegion(res.Get(), 0, up.Get(), 0, size * 4);
        ctx.Sync();
    };

    LoadToGPU(w1, INPUT_SIZE * HIDDEN_SIZE);
    LoadToGPU(b1, HIDDEN_SIZE);
    LoadToGPU(w2, HIDDEN_SIZE * INPUT_SIZE);
    LoadToGPU(b2, INPUT_SIZE);

    std::cout << "[Global::LoadGPUWeights] Success: Continued from last checkpoint." << std::endl;
}
void SaveGPUWeights(DX12Context& ctx, ID3D12Resource* w1, ID3D12Resource* b1, ID3D12Resource* w2, ID3D12Resource* b2, const std::string& filename) {
    if (!w1 || !b1 || !w2 || !b2) return;

    UINT64 sizes[] = {
        (UINT64)INPUT_SIZE * HIDDEN_SIZE * 4,
        (UINT64)HIDDEN_SIZE * 4,
        (UINT64)HIDDEN_SIZE * INPUT_SIZE * 4,
        (UINT64)INPUT_SIZE * 4
    };
    ID3D12Resource* resources[] = { w1, b1, w2, b2 };

    std::ofstream f(filename, std::ios::binary);
    for (int i = 0; i < 4; ++i) {
        auto rb = ctx.CreateReadbackBuf(sizes[i]);
        ctx.list->CopyBufferRegion(rb.Get(), 0, resources[i], 0, sizes[i]);
        ctx.Sync();

        void* p = nullptr;
        D3D12_RANGE range = { 0, sizes[i] };
        if (SUCCEEDED(rb->Map(0, &range, &p))) {
            f.write((char*)p, sizes[i]);
            rb->Unmap(0, nullptr);
        }
    }
    f.close();
    std::cout << "[Global::SaveGPUWeights] Success: Weights saved." << std::endl;
}



void StitchMegaImage(DX12Context& ctx, ID3D12Resource* gpuSource, int tileIdx, int tilesPerRow, std::vector<unsigned char>& outCanvas) {
    uint32_t sz = TILE_SIZE * TILE_SIZE * 4;
    auto rb = ctx.CreateReadbackBuf(sz);

    ctx.list->CopyBufferRegion(rb.Get(), 0, gpuSource, 0, sz);
    ctx.Sync();

    void* p = nullptr;
    rb->Map(0, nullptr, &p);
    float* pixels = (float*)p;

    int tileY = tileIdx / tilesPerRow;
    int tileX = tileIdx % tilesPerRow;
    int canvasWidth = tilesPerRow * TILE_SIZE;

    for (int y = 0; y < TILE_SIZE; ++y) {
        for (int x = 0; x < TILE_SIZE; ++x) {
            int globalX = tileX * TILE_SIZE + x;
            int globalY = tileY * TILE_SIZE + y;
            outCanvas[globalY * canvasWidth + globalX] = (unsigned char)(std::clamp(pixels[y * TILE_SIZE + x], 0.0f, 1.0f) * 255.0f);
        }
    }
    rb->Unmap(0, nullptr);
}


// [Main::GenerateWorld]
void GenerateUniqueWorld(DX12Context& ctx, Pipeline& pFwd, ComPtr<ID3D12Resource> w1, ComPtr<ID3D12Resource> b1,
                         ComPtr<ID3D12Resource> w2, ComPtr<ID3D12Resource> b2,
                         ComPtr<ID3D12Resource> a1, ComPtr<ID3D12Resource> a2,
                         ComPtr<ID3D12Resource> inp, ComPtr<ID3D12Resource> up,
                         std::vector<unsigned char>& canvas) {

    int totalWidth = 1024;
    int tilesPerRow = totalWidth / TILE_SIZE;
    int maxTiles = tilesPerRow * tilesPerRow;

    // Используем генератор случайных чисел для создания уникального входа для каждого тайла
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int t = 0; t < maxTiles; ++t) {
        // 1. Генерируем уникальный шумовой вектор для этого конкретного места на карте
        std::vector<float> noiseInput(INPUT_SIZE);
        for (auto& f : noiseInput) f = dist(gen);

        // 2. Загружаем шум в GPU
        void* pData = nullptr;
        up->Map(0, nullptr, &pData);
        memcpy(pData, noiseInput.data(), INPUT_SIZE * 4);
        up->Unmap(0, nullptr);
        ctx.list->CopyBufferRegion(inp.Get(), 0, up.Get(), 0, INPUT_SIZE * 4);

        // 3. Прогоняем через нейросеть (Forward Pass)
        LayerConfig c1 = { (uint)INPUT_SIZE, (uint)HIDDEN_SIZE, 0.0f };
        ctx.list->SetPipelineState(pFwd.pso.Get());
        ctx.list->SetComputeRootSignature(pFwd.sig.Get());
        ctx.list->SetComputeRoot32BitConstants(0, 4, &c1, 0);
        ctx.list->SetComputeRootShaderResourceView(1, inp->GetGPUVirtualAddress());
        ctx.list->SetComputeRootShaderResourceView(2, w1->GetGPUVirtualAddress());
        ctx.list->SetComputeRootShaderResourceView(3, b1->GetGPUVirtualAddress());
        ctx.list->SetComputeRootUnorderedAccessView(4, a1->GetGPUVirtualAddress());
        ctx.list->Dispatch((HIDDEN_SIZE + 255) / 256, 1, 1);

        D3D12_RESOURCE_BARRIER barrier = { D3D12_RESOURCE_BARRIER_TYPE_UAV };
        barrier.UAV.pResource = a1.Get();
        ctx.list->ResourceBarrier(1, &barrier);

        LayerConfig c2 = { (uint)HIDDEN_SIZE, (uint)INPUT_SIZE, 0.0f };
        ctx.list->SetComputeRoot32BitConstants(0, 4, &c2, 0);
        ctx.list->SetComputeRootShaderResourceView(1, a1->GetGPUVirtualAddress());
        ctx.list->SetComputeRootShaderResourceView(2, w2->GetGPUVirtualAddress());
        ctx.list->SetComputeRootShaderResourceView(3, b2->GetGPUVirtualAddress());
        ctx.list->SetComputeRootUnorderedAccessView(4, a2->GetGPUVirtualAddress());
        ctx.list->Dispatch((INPUT_SIZE + 255) / 256, 1, 1);

        ctx.Sync();

        // 4. Сшиваем результат в полотно
        StitchMegaImage(ctx, a2.Get(), t, tilesPerRow, canvas);


    }
}
std::vector<float> RotateTile(const std::vector<float>& src) {
    std::vector<float> dst(src.size());
    for (int y = 0; y < TILE_SIZE; ++y) {
        for (int x = 0; x < TILE_SIZE; ++x) {
            dst[x * TILE_SIZE + (TILE_SIZE - 1 - y)] = src[y * TILE_SIZE + x];
        }
    }
    return dst;
}

// [Global::PrepareDataset] Нарезка + Аугментация (Повороты)
void PrepareDataset(std::vector<std::vector<float>>& outData) {
    if (!fs::exists("data")) {
        std::cerr << "[Global::PrepareDataset] Error: 'data' folder not found!" << std::endl;
        return;
    }

    for (auto& it : fs::directory_iterator("data")) {
        if (it.path().extension() == ".png") {
            int w, h, ch;
            unsigned char* img = stbi_load(it.path().string().c_str(), &w, &h, &ch, 1);
            if (!img) continue;

            for (int y = 0; y <= h - TILE_SIZE; y += TILE_SIZE) {
                for (int x = 0; x <= w - TILE_SIZE; x += TILE_SIZE) {
                    std::vector<float> tile(INPUT_SIZE);
                    for (int ty = 0; ty < TILE_SIZE; ++ty) {
                        for (int tx = 0; tx < TILE_SIZE; ++tx) {
                            tile[ty * TILE_SIZE + tx] = img[(y + ty) * w + (x + tx)] / 255.0f;
                        }
                    }

                    // Добавляем оригинал
                    outData.push_back(tile);
                    // Добавляем 3 поворота (90, 180, 270)
                    auto r90 = RotateTile(tile);
                    outData.push_back(r90);
                    auto r180 = RotateTile(r90);
                    outData.push_back(r180);
                    auto r270 = RotateTile(r180);
                    outData.push_back(r270);
                }
            }
            stbi_image_free(img);
        }
    }

    // Перемешивание критически важно для обучения
    std::shuffle(outData.begin(), outData.end(), std::mt19937(std::random_device()()));

    std::cout << "[Global::PrepareDataset] Success! Total samples with augmentation: " << outData.size() << std::endl;
}
float CalculateLoss(DX12Context& ctx, ID3D12Resource* output, ID3D12Resource* target, ID3D12Resource* uploadBuf) {
    // Копируем данные выхода и цели в Readback буферы
    auto rbOut = ctx.CreateReadbackBuf(INPUT_SIZE * 4);
    auto rbTar = ctx.CreateReadbackBuf(INPUT_SIZE * 4);

    ctx.list->CopyBufferRegion(rbOut.Get(), 0, output, 0, INPUT_SIZE * 4);
    ctx.list->CopyBufferRegion(rbTar.Get(), 0, target, 0, INPUT_SIZE * 4);
    ctx.Sync();

    float* pOut = nullptr;
    float* pTar = nullptr;
    float mse = 0.0f;

    if (SUCCEEDED(rbOut->Map(0, nullptr, (void**)&pOut)) && SUCCEEDED(rbTar->Map(0, nullptr, (void**)&pTar))) {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            float diff = pOut[i] - pTar[i];
            mse += diff * diff;
        }
        rbOut->Unmap(0, nullptr);
        rbTar->Unmap(0, nullptr);
    }
    return mse / (float)INPUT_SIZE;
}
int main() {
    DX12Context ctx;
    ctx.Init();

    // 1. Выбор режима
    std::cout << "Select mode:\n [1] Train (or continue training)\n [2] Generate 1024x1024 Map\nChoice: ";
    int mode = 0;
    std::cin >> mode;
    bool isTraining = (mode == 1);

    // 2. Подготовка данных (ТОЛЬКО для обучения)
    std::vector<std::vector<float>> data;
    if (isTraining) {
        PrepareDataset(data);
        if (data.empty()) {
            std::cerr << "[Main::Error] No images in 'data/' folder!" << std::endl;
            return -1;
        }
    }

    // 3. Создание PSO (Шейдеры)
    auto pFwd = CreatePSO(&ctx, "CSForward", 3, 1);
    auto pBkO = CreatePSO(&ctx, "CSBackwardOutput", 2, 1);
    auto pBkH = CreatePSO(&ctx, "CSBackwardHidden", 3, 1);
    auto pUpW = CreatePSO(&ctx, "CSUpdateWeights", 2, 1);
    auto pUpB = CreatePSO(&ctx, "CSUpdateBias", 1, 1);

    // 4. Создание БУФЕРОВ (Архитектура остается прежней)
    auto w1 = ctx.CreateBuf(INPUT_SIZE * HIDDEN_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto b1 = ctx.CreateBuf(HIDDEN_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto a1 = ctx.CreateBuf(HIDDEN_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto g1 = ctx.CreateBuf(HIDDEN_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto w2 = ctx.CreateBuf(HIDDEN_SIZE * INPUT_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto b2 = ctx.CreateBuf(INPUT_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto a2 = ctx.CreateBuf(INPUT_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto g2 = ctx.CreateBuf(INPUT_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto inp = ctx.CreateBuf(INPUT_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto tar = ctx.CreateBuf(INPUT_SIZE * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);

    uint32_t maxBufferSize = max(INPUT_SIZE * HIDDEN_SIZE, HIDDEN_SIZE * INPUT_SIZE) * 4;
    auto up = ctx.CreateBuf(maxBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);

    // 5. Загрузка весов
    std::string weightFile = "weights_gpu.bin";
    bool weightsLoaded = false;
    if (std::filesystem::exists(weightFile)) {
        LoadGPUWeights(ctx, w1, b1, w2, b2, up, weightFile);
        weightsLoaded = true;
    } else if (!isTraining) {
        std::cerr << "[Main::Error] weights_gpu.bin not found. Train the model first!" << std::endl;
        return -1;
    } else {
        InitWeights(ctx, w1, up, INPUT_SIZE * HIDDEN_SIZE);
        InitWeights(ctx, w2, up, HIDDEN_SIZE * INPUT_SIZE);
    }

    // Параметры выходного изображения
    const int totalWidth = 1024;
    const int tilesPerRow = totalWidth / TILE_SIZE; // 16
    const int maxTiles = tilesPerRow * tilesPerRow; // 256
    std::vector<unsigned char> canvas(totalWidth * totalWidth, 0);

    if (isTraining) {
        // --- ЛОГИКА ОБУЧЕНИЯ ---
        int epochs = 1;
        std::cout << "Enter epochs: "; std::cin >> epochs;
        uint32_t step = 0;

        for (int e = 0; e < epochs; ++e) {
            for (size_t i = 0; i < data.size(); ++i) {
                // Загрузка примера из датасета
                void* pData = nullptr;
                up->Map(0, nullptr, &pData);
                memcpy(pData, data[i].data(), INPUT_SIZE * 4);
                up->Unmap(0, nullptr);
                ctx.list->CopyBufferRegion(inp.Get(), 0, up.Get(), 0, INPUT_SIZE * 4);
                ctx.list->CopyBufferRegion(tar.Get(), 0, up.Get(), 0, INPUT_SIZE * 4);

                // Forward Pass (W1 -> A1 -> W2 -> A2)
                // ... (тут ваш стандартный Dispatch для pFwd слоя 1 и 2) ...
                // [Здесь оставьте ваш существующий блок Forward/Backward из оригинального кода]

                ctx.Sync();
                step++;
                if (step % 1000 == 0) std::cout << "Step: " << step << " processed." << std::endl;
            }
        }
        SaveGPUWeights(ctx, w1.Get(), b1.Get(), w2.Get(), b2.Get(), weightFile);
    } else {
        // --- ЛОГИКА ГЕНЕРАЦИИ МОНОЛИТНОЙ КАРТЫ ---
        std::cout << "[Main] Generating smooth 1024x1024 map..." << std::endl;

        for (int t = 0; t < maxTiles; ++t) {
            int tileY = t / tilesPerRow;
            int tileX = t % tilesPerRow;

            // Генерируем входной вектор на основе координат и плавных функций
            std::vector<float> noise(INPUT_SIZE);
            for (int y = 0; y < TILE_SIZE; ++y) {
                for (int x = 0; x < TILE_SIZE; ++x) {
                    // Глобальные координаты пикселя в мире 1024x1024
                    float gx = (float)(tileX * TILE_SIZE + x) / totalWidth;
                    float gy = (float)(tileY * TILE_SIZE + y) / totalWidth;

                    // Создаем сложный математический шум (замена Перлину для простоты)
                    // Это создаст плавные переходы между тайлами
                    float val = 0.5f + 0.5f * sin(gx * 10.0f) * cos(gy * 10.0f);
                    val += 0.2f * sin(gx * 50.0f + gy * 20.0f); // Добавляем детализацию

                    noise[y * TILE_SIZE + x] = std::clamp(val, 0.0f, 1.0f);
                }
            }

            // Отправка на GPU
            void* pData = nullptr;
            up->Map(0, nullptr, &pData);
            memcpy(pData, noise.data(), INPUT_SIZE * 4);
            up->Unmap(0, nullptr);
            ctx.list->CopyBufferRegion(inp.Get(), 0, up.Get(), 0, INPUT_SIZE * 4);
            // Forward Pass: Layer 1
            LayerConfig c1 = { (uint)INPUT_SIZE, (uint)HIDDEN_SIZE, 0.0f };
            ctx.list->SetPipelineState(pFwd.pso.Get());
            ctx.list->SetComputeRootSignature(pFwd.sig.Get());
            ctx.list->SetComputeRoot32BitConstants(0, 4, &c1, 0);
            ctx.list->SetComputeRootShaderResourceView(1, inp->GetGPUVirtualAddress());
            ctx.list->SetComputeRootShaderResourceView(2, w1->GetGPUVirtualAddress());
            ctx.list->SetComputeRootShaderResourceView(3, b1->GetGPUVirtualAddress());
            ctx.list->SetComputeRootUnorderedAccessView(4, a1->GetGPUVirtualAddress());
            ctx.list->Dispatch((HIDDEN_SIZE + 255) / 256, 1, 1);

            D3D12_RESOURCE_BARRIER b = { D3D12_RESOURCE_BARRIER_TYPE_UAV };
            b.UAV.pResource = a1.Get();
            ctx.list->ResourceBarrier(1, &b);

            // Forward Pass: Layer 2
            LayerConfig c2 = { (uint)HIDDEN_SIZE, (uint)INPUT_SIZE, 0.0f };
            ctx.list->SetComputeRoot32BitConstants(0, 4, &c2, 0);
            ctx.list->SetComputeRootShaderResourceView(1, a1->GetGPUVirtualAddress());
            ctx.list->SetComputeRootShaderResourceView(2, w2->GetGPUVirtualAddress());
            ctx.list->SetComputeRootShaderResourceView(3, b2->GetGPUVirtualAddress());
            ctx.list->SetComputeRootUnorderedAccessView(4, a2->GetGPUVirtualAddress());
            ctx.list->Dispatch((INPUT_SIZE + 255) / 256, 1, 1);

            ctx.Sync();

            // Сшивка без пропусков: передаем текущий индекс t
            StitchMegaImage(ctx, a2.Get(), t, tilesPerRow, canvas);

            if (t % 32 == 0) std::cout << "Progress: " << (t * 100 / maxTiles) << "%" << std::endl;
        }
    }

    stbi_write_png("final_stitched_world.png", totalWidth, totalWidth, 1, canvas.data(), totalWidth);
    std::cout << "[Main::Success] Done! File saved: final_stitched_world.png" << std::endl;
    return 0;
}