// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// What is right for a tool shipped INSIDE OpenMS and wrong for one that only
/// links it -- ported from DIALibGen, where each item shipped broken once:
///   * TOPPBase prints OpenMS's version when version_ is empty;
///   * OpenMS asks its update server on every start and Qt prints network
///     errors that read as errors from this tool;
///   * -write_ctd / -write_cwl die with "tool does not exist" unless the tool
///     is registered with ToolHandler for the duration of the run.
#pragma once

#include <OpenMS/CONCEPT/VersionInfo.h>
#include <OpenMS/SYSTEM/File.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace dlr
{
  inline std::string verboseVersion()
  {
#ifdef DLR_VERSION
    return std::string(DLR_VERSION) + " (OpenMS " + OpenMS::VersionInfo::getVersion() + ")";
#else
    return OpenMS::VersionInfo::getVersion();
#endif
  }

  /// Set only if the user has not: exporting it yourself wins in both directions.
  inline void disableUpdateCheckUnlessSet()
  {
#ifdef _WIN32
    size_t sz = 0;
    if (getenv_s(&sz, nullptr, 0, "OPENMS_DISABLE_UPDATE_CHECK") != 0 || sz == 0)
    { _putenv_s("OPENMS_DISABLE_UPDATE_CHECK", "ON"); }
#else
    ::setenv("OPENMS_DISABLE_UPDATE_CHECK", "ON", 0);
#endif
  }

  /// Registers one tool with OpenMS's ToolHandler through a temporary .ttd
  /// directory for the lifetime of the object. Never overrides a value the user
  /// set; writes scratch-then-rename so a truncated file can never be found.
  class ToolHandlerRegistration
  {
  public:
    explicit ToolHandlerRegistration(const std::string& tool)
    {
      namespace fs = std::filesystem;
      const char* existing = std::getenv("OPENMS_TTD_INTERNAL_PATH");
      if (existing != nullptr && *existing != '\0') { return; }
      std::error_code ec;
      const fs::path base = fs::temp_directory_path(ec);
      if (ec) { return; }
      const fs::path dir = base / ("dialibrefine-ttd-" + std::string(OpenMS::File::getUniqueName(false)));
      fs::create_directories(dir, ec);
      if (ec || !fs::is_directory(dir, ec)) { return; }
      const std::string ttd = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<ttd><tool status=\"internal\"><name>" + tool + "</name><category/><type/></tool></ttd>\n";
      const fs::path scratch = dir / (tool + ".ttd.part");
      const fs::path target = dir / (tool + ".ttd");
      {
        std::ofstream os(scratch, std::ios::binary | std::ios::trunc);
        os.write(ttd.data(), static_cast<std::streamsize>(ttd.size()));
        os.close();
        if (!os) { fs::remove_all(dir, ec); return; }
      }
      fs::rename(scratch, target, ec);
      if (ec || fs::file_size(target, ec) != ttd.size() || ec) { fs::remove_all(dir, ec); return; }
      const std::string dir_str = dir.string();
#ifdef _WIN32
      const bool set_ok = (_putenv_s("OPENMS_TTD_INTERNAL_PATH", dir_str.c_str()) == 0);
#else
      const bool set_ok = (::setenv("OPENMS_TTD_INTERNAL_PATH", dir_str.c_str(), 1) == 0);
#endif
      if (!set_ok) { fs::remove_all(dir, ec); return; }
      dir_ = dir; active_ = true;
    }
    ~ToolHandlerRegistration()
    {
      if (!active_) { return; }
      std::error_code ec; std::filesystem::remove_all(dir_, ec);
    }
    ToolHandlerRegistration(const ToolHandlerRegistration&) = delete;
    ToolHandlerRegistration& operator=(const ToolHandlerRegistration&) = delete;
  private:
    std::filesystem::path dir_;
    bool active_ = false;
  };
}
