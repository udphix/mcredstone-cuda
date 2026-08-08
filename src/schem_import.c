#include "redstone/world.h"
#include "../third_party/miniz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Sponge Schematic (.schem) importer
 *
 * Format: gzip-compressed NBT with:
 *   - Width/Height/Length (short)
 *   - Palette (compound: block_state_string → int)
 *   - BlockData (byte array, varint-encoded palette indices)
 *
 * We parse the NBT, build the palette, decode BlockData, and place blocks.
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── NBT tag types ────────────────────────────────────────────────────── */
#define NBT_END        0
#define NBT_BYTE       1
#define NBT_SHORT      2
#define NBT_INT        3
#define NBT_LONG       4
#define NBT_FLOAT      5
#define NBT_DOUBLE     6
#define NBT_BYTE_ARRAY 7
#define NBT_STRING     8
#define NBT_LIST       9
#define NBT_COMPOUND   10
#define NBT_INT_ARRAY  11
#define NBT_LONG_ARRAY 12

/* ── NBT reader state ─────────────────────────────────────────────────── */
typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;
} NbtReader;

static int nbt_has(NbtReader* r, size_t n) { return r->pos + n <= r->size; }

static uint8_t nbt_u8(NbtReader* r) {
    if (!nbt_has(r, 1)) return 0;
    return r->data[r->pos++];
}

static int16_t nbt_i16(NbtReader* r) {
    if (!nbt_has(r, 2)) return 0;
    int16_t v = (int16_t)((r->data[r->pos] << 8) | r->data[r->pos + 1]);
    r->pos += 2;
    return v;
}

static int32_t nbt_i32(NbtReader* r) {
    if (!nbt_has(r, 4)) return 0;
    int32_t v = (r->data[r->pos] << 24) | (r->data[r->pos+1] << 16) |
                (r->data[r->pos+2] << 8) | r->data[r->pos+3];
    r->pos += 4;
    return v;
}

static void nbt_skip(NbtReader* r, size_t n) {
    r->pos += n;
    if (r->pos > r->size) r->pos = r->size;
}

static char* nbt_string(NbtReader* r) {
    int16_t len = nbt_i16(r);
    if (len <= 0 || !nbt_has(r, (size_t)len)) return NULL;
    char* s = (char*)malloc(len + 1);
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

/* Skip a tag payload (not including type byte or name) */
static void nbt_skip_payload(NbtReader* r, uint8_t type);

static void nbt_skip_payload(NbtReader* r, uint8_t type) {
    switch (type) {
        case NBT_BYTE:   nbt_skip(r, 1); break;
        case NBT_SHORT:  nbt_skip(r, 2); break;
        case NBT_INT:    nbt_skip(r, 4); break;
        case NBT_LONG:   nbt_skip(r, 8); break;
        case NBT_FLOAT:  nbt_skip(r, 4); break;
        case NBT_DOUBLE: nbt_skip(r, 8); break;
        case NBT_BYTE_ARRAY: {
            int32_t len = nbt_i32(r);
            if (len > 0) nbt_skip(r, len);
            break;
        }
        case NBT_STRING: {
            int16_t len = nbt_i16(r);
            if (len > 0) nbt_skip(r, len);
            break;
        }
        case NBT_LIST: {
            uint8_t elem_type = nbt_u8(r);
            int32_t count = nbt_i32(r);
            for (int32_t i = 0; i < count; i++)
                nbt_skip_payload(r, elem_type);
            break;
        }
        case NBT_COMPOUND: {
            while (r->pos < r->size) {
                uint8_t t = nbt_u8(r);
                if (t == NBT_END) break;
                /* skip name */
                int16_t nlen = nbt_i16(r);
                if (nlen > 0) nbt_skip(r, nlen);
                nbt_skip_payload(r, t);
            }
            break;
        }
        case NBT_INT_ARRAY: {
            int32_t len = nbt_i32(r);
            if (len > 0) nbt_skip(r, (size_t)len * 4);
            break;
        }
        case NBT_LONG_ARRAY: {
            int32_t len = nbt_i32(r);
            if (len > 0) nbt_skip(r, (size_t)len * 8);
            break;
        }
    }
}

/* ── Minecraft block state → our BlockType mapping ────────────────────── */

typedef struct {
    uint8_t type;
    uint8_t facing;
    uint16_t state;
    /* Post-placement fixups (derived from MC state string) */
    uint8_t has_override;   /* if 1, apply overrides below */
    uint8_t ov_signal;      /* target signal value (0-15) */
    uint8_t ov_powered;     /* target POWERED flag (0/1) */
    uint8_t ov_strong;      /* target STRONG_POWER flag (0/1) */
} BlockMapping;

static uint8_t parse_facing(const char* props) {
    const char* f = strstr(props, "facing=");
    if (!f) return DIR_NONE;
    f += 7;
    if (strncmp(f, "north", 5) == 0) return DIR_NORTH;
    if (strncmp(f, "south", 5) == 0) return DIR_SOUTH;
    if (strncmp(f, "east", 4) == 0)  return DIR_EAST;
    if (strncmp(f, "west", 4) == 0)  return DIR_WEST;
    if (strncmp(f, "up", 2) == 0)    return DIR_UP;
    if (strncmp(f, "down", 4) == 0)  return DIR_DOWN;
    return DIR_NONE;
}

static uint16_t parse_delay(const char* props) {
    const char* d = strstr(props, "delay=");
    if (!d) return 1;
    return (uint16_t)(d[6] - '0');
}

static int prop_bool(const char* props, const char* key) {
    /* Return 1 if key=true, 0 otherwise (false or absent). */
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s=true", key);
    (void)n;
    return strstr(props, buf) != NULL;
}

static int parse_power(const char* props) {
    /* Parse 'power=N' (0-15). Returns -1 if missing. */
    const char* p = strstr(props, "power=");
    if (!p) return -1;
    p += 6;
    int v = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
    if (v < 0) v = 0;
    if (v > 15) v = 15;
    return v;
}

static BlockMapping map_block_state(const char* block_state) {
    BlockMapping m = { BLOCK_AIR, DIR_NONE, 0 };

    /* Find the block name (before '[' if properties exist) */
    const char* props = strchr(block_state, '[');

    /* Strip "minecraft:" prefix */
    const char* name = block_state;
    if (strncmp(name, "minecraft:", 10) == 0) name += 10;

    /* Match block names */
    if (strncmp(name, "air", 3) == 0 || strncmp(name, "cave_air", 8) == 0) {
        m.type = BLOCK_AIR;
    }
    else if (strncmp(name, "redstone_wire", 13) == 0) {
        m.type = BLOCK_REDSTONE_DUST;
        if (props) {
            int pwr = parse_power(props);
            if (pwr >= 0) {
                m.has_override = 1;
                m.ov_signal = (uint8_t)pwr;
                m.ov_powered = pwr > 0;
            }
        }
    }
    else if (strncmp(name, "redstone_wall_torch", 19) == 0) {
        m.type = BLOCK_REDSTONE_TORCH;
        if (props) {
            /* MC facing = direction torch FACES. Our facing = direction of ATTACHED block.
             * These are OPPOSITE: MC facing=north means attached to block to the SOUTH. */
            uint8_t mc_facing = parse_facing(props);
            const uint8_t opposite[] = { 1, 0, 3, 2, 5, 4, 6 };
            m.facing = opposite[mc_facing < 7 ? mc_facing : 6];
            int lit = prop_bool(props, "lit");
            m.has_override = 1;
            m.ov_signal = lit ? 15 : 0;
            m.ov_powered = lit ? 1 : 0;
        }
    }
    else if (strncmp(name, "redstone_torch", 14) == 0) {
        m.type = BLOCK_REDSTONE_TORCH;
        m.facing = DIR_DOWN; /* floor torch */
        if (props) {
            int lit = prop_bool(props, "lit");
            m.has_override = 1;
            m.ov_signal = lit ? 15 : 0;
            m.ov_powered = lit ? 1 : 0;
        } else {
            /* No props → assume lit (MC default) */
            m.has_override = 1;
            m.ov_signal = 15;
            m.ov_powered = 1;
        }
    }
    else if (strncmp(name, "redstone_block", 14) == 0) {
        m.type = BLOCK_REDSTONE_BLOCK;
    }
    else if (strncmp(name, "redstone_lamp", 13) == 0) {
        m.type = BLOCK_LAMP;
        if (props) {
            int lit = prop_bool(props, "lit");
            m.has_override = 1;
            m.ov_signal = lit ? 15 : 0;
            m.ov_powered = lit ? 1 : 0;
        }
    }
    else if (strncmp(name, "repeater", 8) == 0) {
        m.type = BLOCK_REPEATER;
        if (props) {
            /* MC convention: FACING = INPUT direction (back of repeater).
             *   - updateNeighborsInFront uses pos.relative(FACING.opposite)
             *     to reach the OUTPUT side (DiodeBlock.java line 185).
             *   - getInputSignal reads from pos.relative(FACING) (line 135).
             * Our sim convention: facing = OUTPUT direction (signal flows
             * toward facing). So we must INVERT MC's facing on import. */
            const uint8_t opposite[] = { 1, 0, 3, 2, 5, 4, 6 };
            uint8_t mc_facing = parse_facing(props);
            m.facing = opposite[mc_facing < 7 ? mc_facing : 6];
            m.state = parse_delay(props);
            int powered = prop_bool(props, "powered");
            m.has_override = 1;
            m.ov_signal = powered ? 15 : 0;
            m.ov_powered = powered ? 1 : 0;
            m.ov_strong = powered ? 1 : 0;
        }
    }
    else if (strncmp(name, "comparator", 10) == 0) {
        m.type = BLOCK_COMPARATOR;
        if (props) {
            /* Same convention flip as repeater (MC FACING = input). */
            const uint8_t opposite[] = { 1, 0, 3, 2, 5, 4, 6 };
            uint8_t mc_facing = parse_facing(props);
            m.facing = opposite[mc_facing < 7 ? mc_facing : 6];

            /* mode=compare|subtract → state bit 0. */
            if (strstr(props, "mode=subtract") != NULL) m.state |= 0x1;

            /* MC only stores a boolean `powered`, not the output level. A lit
             * comparator imports as 15; the first tile tick recomputes the
             * real level from its inputs. */
            int powered = prop_bool(props, "powered");
            m.has_override = 1;
            m.ov_signal = powered ? 15 : 0;
            m.ov_strong = powered ? 1 : 0;
        }
    }
    else if (strncmp(name, "lever", 5) == 0) {
        m.type = BLOCK_LEVER;
        if (props) {
            /* MC lever has face=floor/wall/ceiling and facing=north/south/etc.
             * face=floor → attached to block below (our facing=DIR_DOWN)
             * face=ceiling → attached to block above (our facing=DIR_UP)
             * face=wall → facing is direction lever faces, attached is OPPOSITE */
            if (strstr(props, "face=floor")) {
                m.facing = DIR_DOWN;
            } else if (strstr(props, "face=ceiling")) {
                m.facing = DIR_UP;
            } else {
                /* Wall lever: MC facing = direction lever faces, invert for our convention */
                uint8_t mc_facing = parse_facing(props);
                const uint8_t opposite[] = { 1, 0, 3, 2, 5, 4, 6 };
                m.facing = opposite[mc_facing < 7 ? mc_facing : 6];
            }
            int powered = prop_bool(props, "powered");
            m.has_override = 1;
            m.ov_signal = powered ? 15 : 0;
            m.ov_powered = powered ? 1 : 0;
            m.ov_strong = powered ? 1 : 0;
        }
    }
    else if (strncmp(name, "stone_button", 12) == 0 ||
             strncmp(name, "oak_button", 10) == 0) {
        m.type = BLOCK_BUTTON;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "piston", 6) == 0 && strncmp(name, "piston_head", 11) != 0) {
        m.type = BLOCK_PISTON;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "sticky_piston", 13) == 0) {
        m.type = BLOCK_STICKY_PISTON;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "observer", 8) == 0) {
        m.type = BLOCK_OBSERVER;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "dropper", 7) == 0) {
        m.type = BLOCK_DROPPER;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "dispenser", 9) == 0) {
        m.type = BLOCK_DISPENSER;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "hopper", 6) == 0) {
        m.type = BLOCK_HOPPER;
        if (props) m.facing = parse_facing(props);
    }
    else if (strncmp(name, "tnt", 3) == 0) {
        m.type = BLOCK_TNT;
    }
    else if (strncmp(name, "note_block", 10) == 0) {
        m.type = BLOCK_NOTE_BLOCK;
    }
    else if (strncmp(name, "target", 6) == 0) {
        m.type = BLOCK_TARGET;
    }
    else {
        /* Unknown block → treat as solid if not air-like */
        if (strstr(name, "glass") || strstr(name, "slab") ||
            strstr(name, "stair") || strstr(name, "wall") ||
            strstr(name, "fence") || strstr(name, "sign") ||
            strstr(name, "door") || strstr(name, "trapdoor") ||
            strstr(name, "pressure_plate")) {
            m.type = BLOCK_AIR; /* transparent/non-conducting */
        } else {
            m.type = BLOCK_SOLID;
        }
    }

    return m;
}

/* ── Read varint from BlockData ───────────────────────────────────────── */
static int32_t read_varint(const uint8_t* data, size_t size, size_t* pos) {
    int32_t value = 0;
    int shift = 0;
    while (*pos < size) {
        uint8_t b = data[(*pos)++];
        value |= (int32_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift >= 32) break;
    }
    return value;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Main .schem import function
 * ══════════════════════════════════════════════════════════════════════════ */

int world_import_schem(World* world, const char* path) {
    /* Read the gzipped file */
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Schem] Cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* compressed = (uint8_t*)malloc(file_size);
    fread(compressed, 1, file_size, f);
    fclose(f);

    /* Decompress gzip → raw NBT */
    size_t decomp_size = (size_t)file_size * 16;
    uint8_t* decompressed = (uint8_t*)malloc(decomp_size);
    mz_ulong actual_size = (mz_ulong)decomp_size;

    /* Skip gzip header (10 bytes) and inflate raw deflate stream */
    size_t data_start = 0;
    if (file_size >= 10 && compressed[0] == 0x1F && compressed[1] == 0x8B) {
        /* Gzip format — skip header */
        data_start = 10;
        /* If FEXTRA flag set, skip extra field */
        uint8_t flags = compressed[3];
        if (flags & 0x04) { /* FEXTRA */
            if (data_start + 2 <= (size_t)file_size) {
                uint16_t xlen = compressed[data_start] | (compressed[data_start+1] << 8);
                data_start += 2 + xlen;
            }
        }
        if (flags & 0x08) { /* FNAME */
            while (data_start < (size_t)file_size && compressed[data_start]) data_start++;
            data_start++;
        }
        if (flags & 0x10) { /* FCOMMENT */
            while (data_start < (size_t)file_size && compressed[data_start]) data_start++;
            data_start++;
        }
        if (flags & 0x02) data_start += 2; /* FHCRC */
    }

    /* Raw inflate (no header, -15 window bits) */
    mz_stream stream;
    memset(&stream, 0, sizeof(stream));
    stream.next_in = compressed + data_start;
    stream.avail_in = (unsigned int)(file_size - data_start);
    stream.next_out = decompressed;
    stream.avail_out = (unsigned int)decomp_size;

    int ret;
    if (mz_inflateInit2(&stream, -MZ_DEFAULT_WINDOW_BITS) != MZ_OK) {
        fprintf(stderr, "[Schem] Failed to init inflate\n");
        free(compressed); free(decompressed);
        return -1;
    }
    ret = mz_inflate(&stream, MZ_FINISH);
    actual_size = (mz_ulong)stream.total_out;
    mz_inflateEnd(&stream);

    if (ret != MZ_STREAM_END && ret != MZ_OK) {
        fprintf(stderr, "[Schem] Decompression failed (%d)\n", ret);
        free(compressed); free(decompressed);
        return -1;
    }

    free(compressed);
    printf("[Schem] Decompressed %ld → %lu bytes\n", file_size, (unsigned long)actual_size);

    /* Parse NBT to extract schematic data */
    NbtReader r = { decompressed, (size_t)actual_size, 0 };

    /* Root tag should be a compound */
    uint8_t root_type = nbt_u8(&r);
    if (root_type != NBT_COMPOUND) {
        fprintf(stderr, "[Schem] Expected root compound, got %d\n", root_type);
        free(decompressed);
        return -1;
    }
    char* root_name = nbt_string(&r);
    if (root_name) free(root_name);

    /* State variables we're looking for */
    int16_t width = 0, height = 0, length = 0;

    #define MAX_PALETTE 4096
    char* palette_names[MAX_PALETTE];
    int palette_ids[MAX_PALETTE];
    int palette_count = 0;

    uint8_t* block_data = NULL;
    int32_t block_data_len = 0;

    /* Parse the root compound */
    int depth = 0;
    while (r.pos < r.size) {
        uint8_t tag_type = nbt_u8(&r);
        if (tag_type == NBT_END) {
            if (depth > 0) { depth--; continue; }
            break;
        }

        char* tag_name = nbt_string(&r);
        if (!tag_name) break;

        /* Check for Sponge Schematic v3 "Schematic" wrapper compound */
        if (tag_type == NBT_COMPOUND && strcmp(tag_name, "Schematic") == 0) {
            free(tag_name);
            depth++;
            continue; /* Enter this compound, keep parsing */
        }

        if (tag_type == NBT_SHORT && strcmp(tag_name, "Width") == 0) {
            width = nbt_i16(&r);
        }
        else if (tag_type == NBT_SHORT && strcmp(tag_name, "Height") == 0) {
            height = nbt_i16(&r);
        }
        else if (tag_type == NBT_SHORT && strcmp(tag_name, "Length") == 0) {
            length = nbt_i16(&r);
        }
        else if (tag_type == NBT_COMPOUND &&
                 (strcmp(tag_name, "Palette") == 0 ||
                  strcmp(tag_name, "BlockPalette") == 0)) {
            /* Read palette: string → int mappings */
            while (r.pos < r.size) {
                uint8_t pt = nbt_u8(&r);
                if (pt == NBT_END) break;
                char* pname = nbt_string(&r);
                if (!pname) break;
                if (pt == NBT_INT && palette_count < MAX_PALETTE) {
                    palette_names[palette_count] = pname;
                    palette_ids[palette_count] = nbt_i32(&r);
                    palette_count++;
                } else {
                    nbt_skip_payload(&r, pt);
                    free(pname);
                }
            }
        }
        else if (tag_type == NBT_BYTE_ARRAY &&
                 (strcmp(tag_name, "BlockData") == 0 ||
                  strcmp(tag_name, "Data") == 0)) {
            block_data_len = nbt_i32(&r);
            if (block_data_len > 0 && nbt_has(&r, block_data_len)) {
                block_data = (uint8_t*)malloc(block_data_len);
                memcpy(block_data, r.data + r.pos, block_data_len);
                r.pos += block_data_len;
            }
        }
        else if (tag_type == NBT_COMPOUND &&
                 strcmp(tag_name, "Blocks") == 0) {
            /* Sponge v3: Blocks compound contains Palette and Data */
            free(tag_name);
            depth++;
            continue;
        }
        else {
            nbt_skip_payload(&r, tag_type);
        }

        free(tag_name);
    }

    printf("[Schem] Dimensions: %dx%dx%d, Palette: %d entries, BlockData: %d bytes\n",
           width, height, length, palette_count, block_data_len);

    if (width == 0 || height == 0 || length == 0 || !block_data || palette_count == 0) {
        fprintf(stderr, "[Schem] Missing required data\n");
        for (int i = 0; i < palette_count; i++) free(palette_names[i]);
        if (block_data) free(block_data);
        free(decompressed);
        return -1;
    }

    /* Build palette index → BlockMapping lookup */
    int max_palette_id = 0;
    for (int i = 0; i < palette_count; i++) {
        if (palette_ids[i] > max_palette_id) max_palette_id = palette_ids[i];
    }

    BlockMapping* mappings = (BlockMapping*)calloc(max_palette_id + 1, sizeof(BlockMapping));
    for (int i = 0; i < palette_count; i++) {
        BlockMapping bm = map_block_state(palette_names[i]);
        if (palette_ids[i] <= max_palette_id) {
            mappings[palette_ids[i]] = bm;
        }
    }

    /* Decode BlockData (varint) and place blocks */
    size_t bd_pos = 0;
    int blocks_placed = 0;

    for (int y = 0; y < height; y++) {
        for (int z = 0; z < length; z++) {
            for (int x = 0; x < width; x++) {
                if (bd_pos >= (size_t)block_data_len) goto done;

                int32_t palette_idx = read_varint(block_data, block_data_len, &bd_pos);
                if (palette_idx < 0 || palette_idx > max_palette_id) continue;

                BlockMapping* bm = &mappings[palette_idx];
                if (bm->type == BLOCK_AIR) continue;

                if (world_in_bounds(world, x, y, z)) {
                    world_set_block(world, x, y, z,
                                    (BlockType)bm->type,
                                    (Direction)bm->facing,
                                    bm->state);
                    /* Apply state overrides (powered/lit/power from MC props) */
                    if (bm->has_override) {
                        int bidx = world_index(world, x, y, z);
                        Block* blk = &world->h_blocks[bidx];
                        blk->signal = bm->ov_signal;
                        uint8_t flags = 0;
                        if (bm->ov_powered) flags |= BLOCK_FLAG_POWERED;
                        if (bm->ov_strong)  flags |= BLOCK_FLAG_STRONG_POWER;
                        blk->flags = flags;
                    }
                    blocks_placed++;
                }
            }
        }
    }

done:
    printf("[Schem] Placed %d blocks\n", blocks_placed);

    /* Cleanup */
    for (int i = 0; i < palette_count; i++) free(palette_names[i]);
    free(mappings);
    free(block_data);
    free(decompressed);

    return blocks_placed;
}
