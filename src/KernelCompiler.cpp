#include "KernelCompiler.h"
#include "Context.h"
#include "Sha3Hash.h"
#include "log.h"
#include "timeutil.h"
#include "Args.h"

#include <cassert>
#include <cinttypes>
#include <future>

using namespace std;

// Implemented in bundle.cpp
const std::vector<const char*>& getClFileNames();
const std::vector<const char*>& getClFiles();

static_assert(sizeof(Program) == sizeof(cl_program));

// -cl-fast-relaxed-math  -cl-unsafe-math-optimizations -cl-denorms-are-zero -cl-mad-enable
// Other options:
// * -cl-uniform-work-group-size
// * -fno-bin-llvmir
// * various: -fno-bin-source -fno-bin-amdil

KernelCompiler::KernelCompiler(const Args& args, const Context* context, const string& clArgs) :
  cacheDir{args.cacheDir.string()},
  context{context->get()},
  linkArgs{"-cl-finite-math-only " },
  baseArgs{linkArgs + "-cl-std=CL2.0 " + clArgs},
  dump{args.dump},
  useCache{args.useCache},
  verbose{args.verbose},
  deviceId{context->deviceId()}
{

  string hw = getDriverVersion(deviceId) + ':' + getDeviceName(deviceId);
  if (args.verbose) { log("OpenCL: %s, args %s\n", hw.c_str(), baseArgs.c_str()); }

  SHA3 hasher;
  hasher.update(hw);
  hasher.update(baseArgs);

  auto& clNames = getClFileNames();
  auto& clFiles = getClFiles();
  assert(clNames.size() == clFiles.size());
  int n = clNames.size();
  for (int i = 0; i < n; ++i) {
    auto &src = clFiles[i];
    files.push_back({clNames[i], src});
    clSources.push_back(loadSource(context->get(), src));

    hasher.update(clNames[i]);
    hasher.update(src);
  }
  contextHash = std::move(hasher).finish()[0];
  // log("OpenCL %d files, hash %016" PRIx64 "\n", n, contextHash);
}

Program KernelCompiler::compile(const string& fileName, const string& extraArgs) const {
  Program p1 = loadSource(context, "#include \""s + fileName + "\"\n");
  assert(p1);
  
  string args = baseArgs + ' ' + extraArgs;
  if (!dump.empty()) {
    args += " -save-temps="s + dump + "/" + fileName;
  }
  int err = clCompileProgram(p1.get(), 1, &deviceId, args.c_str(),
                             clSources.size(), (const cl_program*) (clSources.data()), getClFileNames().data(),
                             nullptr, nullptr);
  if (string mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Compiling '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), args.c_str());
    return {};
  }
  
  Program p2{clLinkProgram(context, 1, &deviceId, linkArgs.c_str(),
                           1, (cl_program *) &p1, nullptr, nullptr, &err)};
  if (string mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Linking '%s' error %s (args %s)\n", fileName.c_str(), errMes(err).c_str(), linkArgs.c_str());
  }
  return p2;
}

static string to_hex(u64 d) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%016" PRIx64, d);
  return buf;
}

KernelHolder KernelCompiler::loadAux(const string& fileName, const string& kernelName, const string& args) const {
  Timer timer;
  bool fromCache = true;

  Program program;
  string cacheFile;

  if (useCache) {
    string f = kernelName + '-' + to_hex(SHA3::hash(contextHash, fileName, kernelName, args)[0]);
    cacheFile = cacheDir + '/' + f;
    program = loadBinary(context, deviceId, cacheFile);
  }

  if (!program) {
    fromCache = false;
    program = compile(fileName, args);
  }

  if (!program) {
    log("Can't compile %s\n", fileName.c_str());
    throw "Can't compile " + fileName;
  }

  KernelHolder ret{loadKernel(program.get(), kernelName.c_str())};
  if (!ret) {
    log("Can't find %s in %s\n", kernelName.c_str(), fileName.c_str());
    throw "Can't find "s + kernelName + " in " + fileName;
  }

  if (!fromCache) {
    if (useCache) {
      if (verbose) { log("saving binary to '%s'\n", cacheFile.c_str()); }
      saveBinary(program.get(), cacheFile);
    }
    if (verbose) { log("Loaded %s %s: %.0fms\n", kernelName.c_str(), args.c_str(), timer.at() * 1000); }
  }

  return ret;
}

std::future<KernelHolder> KernelCompiler::load(const string& fileName, const string& kernelName, const string& args) const {
#if 0
  // Do the compilation in parallel on a separate thread.
  // Unfortunatelly no benefit on ROCm (the compiler serializes).
  return async(std::launch::async, &KernelCompiler::loadAux, this, fileName, kernelName, args);
#else
  std::promise<KernelHolder> promise;
  promise.set_value(loadAux(fileName, kernelName, args));
  return promise.get_future();
#endif
}
