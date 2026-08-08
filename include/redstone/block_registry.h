#ifndef REDSTONE_BLOCK_REGISTRY_H
#define REDSTONE_BLOCK_REGISTRY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Block Registry ───────────────────────────────────────────────────────
 * Data-driven definitions for all block types.
 * To add a new block: add entry in block_registry.c, write its kernel.
 */

void             registry_init(void);
const BlockDef*  registry_get(BlockType type);
const char*      registry_name(BlockType type);
int              registry_conducts_power(BlockType type);
int              registry_is_power_source(BlockType type);
int              registry_is_transparent(BlockType type);

#ifdef __cplusplus
}
#endif

#endif /* REDSTONE_BLOCK_REGISTRY_H */
