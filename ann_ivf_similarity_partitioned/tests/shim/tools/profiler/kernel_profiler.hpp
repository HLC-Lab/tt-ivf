#pragma once
#ifdef IVF_SCOPE_TEST
// The actual kernel creates/destructs these at exactly the production scope
// boundaries. Transport tests retain the no-op shim below.
#define DeviceZoneScopedN(name) ScopeRecord zone(name)
#else
#define DeviceZoneScopedN(name) do {} while(false)
#endif
