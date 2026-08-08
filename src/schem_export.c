#include "redstone/world.h"
#include "../third_party/miniz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Sponge Schematic (.schem) exporter — v2 format
 *
 * Writes: gzip(NBT) with Palette, BlockData, Width/Height/Length
 * Compatible with Litematica, WorldEdit, and other tools.
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── NBT writer ───────────────────────────────────────────────────────── */

typedef struct {
    uint8_t* data;
    size_t   size;
    size_t   capacity;
} NbtWriter;

static void nbt_init(NbtWriter* w) {
    w->capacity = 4096;
    w->data = (uint8_t*)malloc(w->capacity);
    w->size = 0;
}

static void nbt_ensure(NbtWriter* w, size_t need) {
    while (w->size + need > w->capacity) {
        w->capacity *= 2;
        w->data = (uint8_t*)realloc(w->data, w->capacity);
    }
}

static void nbt_write_u8(NbtWriter* w, uint8_t v) {
    nbt_ensure(w, 1);
    w->data[w->size++] = v;
}

static void nbt_write_i16(NbtWriter* w, int16_t v) {
    nbt_ensure(w, 2);
    w->data[w->size++] = (uint8_t)(v >> 8);
    w->data[w->size++] = (uint8_t)(v & 0xFF);
}

static void nbt_write_i32(NbtWriter* w, int32_t v) {
    nbt_ensure(w, 4);
    w->data[w->size++] = (uint8_t)((v >> 24) & 0xFF);
    w->data[w->size++] = (uint8_t)((v >> 16) & 0xFF);
    w->data[w->size++] = (uint8_t)((v >> 8) & 0xFF);
    w->data[w->size++] = (uint8_t)(v & 0xFF);
}

static void nbt_write_string(NbtWriter* w, const char* s) {
    int16_t len = (int16_t)strlen(s);
    nbt_write_i16(w, len);
    nbt_ensure(w, len);
    memcpy(w->data + w->size, s, len);
    w->size += len;
}

static void nbt_write_bytes(NbtWriter* w, const uint8_t* data, size_t len) {
    nbt_ensure(w, len);
    memcpy(w->data + w->size, data, len);
    w->size += len;
}

/* Tag header: type + name */
static void nbt_tag_header(NbtWriter* w, uint8_t type, const char* name) {
    nbt_write_u8(w, type);
    nbt_write_string(w, name);
}

#define NBT_COMPOUND   10
#define NBT_SHORT      2
#define NBT_INT        3
#define NBT_BYTE_ARRAY 7
#define NBT_STRING     8
#define NBT_END        0

/* ── Block state string generation ────────────────────────────────────── */

static const char* facing_to_mc(uint8_t facing) {
    switch (facing) {
        case 0: return "down";
        case 1: return "up";
        case 2: return "north";
        case 3: return "south";
        case 4: return "west";
        case 5: return "east";
        default: return "north";
    }
}

static void block_to_state_string(char* buf, size_t bufsize, const Block* block) {
    switch (block->type) {
        case 0: /* AIR */
            snprintf(buf, bufsize, "minecraft:air");
            break;
        case 1: /* SOLID */
            snprintf(buf, bufsize, "minecraft:stone");
            break;
        case 10: /* LEVER */ {
            /* Our `facing` = direction of the ATTACHED block (the one the
             * lever is stuck to). In MC:
             *   face=floor    → attached to block BELOW (our DIR_DOWN = 0)
             *   face=ceiling  → attached to block ABOVE (our DIR_UP = 1)
             *   face=wall     → attached to a horizontal neighbor
             * For wall levers, MC's `facing` = direction from wall TOWARD
             * the lever. If attached-block is east (our DIR_EAST=5), then
             * the lever is WEST of that block → MC `facing=west`.
             */
            const char* mc_face;
            const char* mc_facing;
            switch (block->facing) {
                case 0: /* DIR_DOWN — lever on top of floor */
                    mc_face = "floor"; mc_facing = "north"; break;
                case 1: /* DIR_UP — lever on ceiling */
                    mc_face = "ceiling"; mc_facing = "north"; break;
                case 2: /* DIR_NORTH — attached-block is north */
                    mc_face = "wall"; mc_facing = "south"; break;
                case 3: /* DIR_SOUTH */
                    mc_face = "wall"; mc_facing = "north"; break;
                case 4: /* DIR_WEST */
                    mc_face = "wall"; mc_facing = "east"; break;
                case 5: /* DIR_EAST */
                    mc_face = "wall"; mc_facing = "west"; break;
                default:
                    mc_face = "floor"; mc_facing = "north"; break;
            }
            snprintf(buf, bufsize,
                "minecraft:lever[face=%s,facing=%s,powered=%s]",
                mc_face, mc_facing,
                block->signal > 0 ? "true" : "false");
            break;
        }
        case 11: /* BUTTON */
            snprintf(buf, bufsize, "minecraft:stone_button[face=wall,facing=%s]",
                     facing_to_mc(block->facing));
            break;
        case 13: /* REDSTONE_BLOCK */
            snprintf(buf, bufsize, "minecraft:redstone_block");
            break;
        case 20: /* REDSTONE_DUST */
            snprintf(buf, bufsize,
                "minecraft:redstone_wire[east=side,north=side,power=%d,south=side,west=side]",
                block->signal);
            break;
        case 30: /* REDSTONE_TORCH */
            if (block->facing == 0) { /* DIR_DOWN = on floor */
                snprintf(buf, bufsize, "minecraft:redstone_torch[lit=%s]",
                         block->signal > 0 ? "true" : "false");
            } else {
                /* Our facing = direction of attached block.
                 * MC facing = direction torch FACES (opposite of attached).
                 * WEST attached → torch faces EAST, etc. */
                static const char* opposite_face[] = {
                    "up","down","south","north","east","west","north"
                };
                snprintf(buf, bufsize, "minecraft:redstone_wall_torch[facing=%s,lit=%s]",
                         opposite_face[block->facing < 7 ? block->facing : 6],
                         block->signal > 0 ? "true" : "false");
            }
            break;
        case 31: /* REPEATER */ {
            int delay = block->state & 0xF;
            if (delay < 1) delay = 1;
            snprintf(buf, bufsize, "minecraft:repeater[delay=%d,facing=%s,powered=%s]",
                     delay, facing_to_mc(block->facing),
                     block->signal > 0 ? "true" : "false");
            break;
        }
        case 32: /* COMPARATOR */
            snprintf(buf, bufsize, "minecraft:comparator[facing=%s,mode=compare]",
                     facing_to_mc(block->facing));
            break;
        case 40: /* PISTON */
            snprintf(buf, bufsize, "minecraft:piston[facing=%s]",
                     facing_to_mc(block->facing));
            break;
        case 41: /* STICKY_PISTON */
            snprintf(buf, bufsize, "minecraft:sticky_piston[facing=%s]",
                     facing_to_mc(block->facing));
            break;
        case 54: /* LAMP */
            snprintf(buf, bufsize, "minecraft:redstone_lamp[lit=%s]",
                     (block->flags & 0x01) ? "true" : "false");
            break;
        default:
            snprintf(buf, bufsize, "minecraft:stone");
            break;
    }
}

/* ── Varint encoding ──────────────────────────────────────────────────── */

static size_t write_varint(uint8_t* buf, int32_t value) {
    size_t written = 0;
    uint32_t uval = (uint32_t)value;
    do {
        uint8_t b = (uint8_t)(uval & 0x7F);
        uval >>= 7;
        if (uval > 0) b |= 0x80;
        buf[written++] = b;
    } while (uval > 0);
    return written;
}

/* ══════════════════════════════════════════════════════════════════════════ */

int world_export_schem(const World* world, const char* path) {
    /* First pass: build palette from all non-air blocks */
    #define MAX_PALETTE 4096
    char palette_strings[MAX_PALETTE][128];
    int palette_count = 0;

    /* Always have air at index 0 */
    snprintf(palette_strings[0], 128, "minecraft:air");
    palette_count = 1;

    /* Find bounding box of non-air blocks */
    int min_x = world->size_x, max_x = -1;
    int min_y = world->size_y, max_y = -1;
    int min_z = world->size_z, max_z = -1;

    for (int y = 0; y < world->size_y; y++)
        for (int z = 0; z < world->size_z; z++)
            for (int x = 0; x < world->size_x; x++) {
                Block b = world->h_blocks[world_index(world, x, y, z)];
                if (b.type != 0) {
                    if (x < min_x) min_x = x; if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y; if (y > max_y) max_y = y;
                    if (z < min_z) min_z = z; if (z > max_z) max_z = z;
                }
            }

    if (max_x < 0) {
        fprintf(stderr, "[Schem] No blocks to export\n");
        return -1;
    }

    int width  = max_x - min_x + 1;
    int height = max_y - min_y + 1;
    int length = max_z - min_z + 1;

    /* Map each block position to a palette index */
    int total = width * height * length;
    int* block_indices = (int*)calloc(total, sizeof(int)); /* 0 = air */

    for (int y = 0; y < height; y++)
        for (int z = 0; z < length; z++)
            for (int x = 0; x < width; x++) {
                Block b = world->h_blocks[world_index(world,
                    x + min_x, y + min_y, z + min_z)];

                if (b.type == 0) continue; /* air → index 0 */

                char state_str[128];
                block_to_state_string(state_str, sizeof(state_str), &b);

                /* Find or add to palette */
                int found = -1;
                for (int p = 0; p < palette_count; p++) {
                    if (strcmp(palette_strings[p], state_str) == 0) {
                        found = p;
                        break;
                    }
                }
                if (found < 0 && palette_count < MAX_PALETTE) {
                    found = palette_count;
                    strncpy(palette_strings[palette_count], state_str, 127);
                    palette_strings[palette_count][127] = '\0';
                    palette_count++;
                }

                int idx = (y * length + z) * width + x;
                block_indices[idx] = (found >= 0) ? found : 0;
            }

    /* Encode BlockData as varints */
    uint8_t* block_data = (uint8_t*)malloc(total * 5); /* max 5 bytes per varint */
    size_t bd_size = 0;
    for (int i = 0; i < total; i++) {
        bd_size += write_varint(block_data + bd_size, block_indices[i]);
    }

    free(block_indices);

    /* Build NBT — Sponge Schematic v2 format
     * Structure: root compound "" { Version, DataVersion, Width, Height,
     *            Length, Palette{}, PaletteMax, BlockData, Offset[] }  */
    NbtWriter nbt;
    nbt_init(&nbt);

    #define NBT_INT_ARRAY 11

    /* Root compound — empty name per NBT spec */
    nbt_tag_header(&nbt, NBT_COMPOUND, "Schematic");

    /* Required: Version */
    nbt_tag_header(&nbt, NBT_INT, "Version");
    nbt_write_i32(&nbt, 2);

    /* Required: DataVersion (MC 1.20.4 = 3700) */
    nbt_tag_header(&nbt, NBT_INT, "DataVersion");
    nbt_write_i32(&nbt, 3700);

    /* Required: Dimensions */
    nbt_tag_header(&nbt, NBT_SHORT, "Width");
    nbt_write_i16(&nbt, (int16_t)width);
    nbt_tag_header(&nbt, NBT_SHORT, "Height");
    nbt_write_i16(&nbt, (int16_t)height);
    nbt_tag_header(&nbt, NBT_SHORT, "Length");
    nbt_write_i16(&nbt, (int16_t)length);

    /* Required: PaletteMax */
    nbt_tag_header(&nbt, NBT_INT, "PaletteMax");
    nbt_write_i32(&nbt, palette_count);

    /* Required: Palette (compound of string→int) */
    nbt_tag_header(&nbt, NBT_COMPOUND, "Palette");
    for (int i = 0; i < palette_count; i++) {
        nbt_tag_header(&nbt, NBT_INT, palette_strings[i]);
        nbt_write_i32(&nbt, i);
    }
    nbt_write_u8(&nbt, NBT_END);

    /* Required: BlockData (byte array, varint encoded) */
    nbt_tag_header(&nbt, NBT_BYTE_ARRAY, "BlockData");
    nbt_write_i32(&nbt, (int32_t)bd_size);
    nbt_write_bytes(&nbt, block_data, bd_size);

    /* BlockEntities — empty list of compounds (Litematica expects this) */
    #define NBT_LIST 9
    nbt_tag_header(&nbt, NBT_LIST, "BlockEntities");
    nbt_write_u8(&nbt, NBT_COMPOUND); /* element type */
    nbt_write_i32(&nbt, 0);           /* count = 0 */

    /* Offset (int array [0,0,0]) */
    nbt_tag_header(&nbt, NBT_INT_ARRAY, "Offset");
    nbt_write_i32(&nbt, 3);
    nbt_write_i32(&nbt, 0);
    nbt_write_i32(&nbt, 0);
    nbt_write_i32(&nbt, 0);

    /* Metadata (empty compound) */
    nbt_tag_header(&nbt, NBT_COMPOUND, "Metadata");
    nbt_write_u8(&nbt, NBT_END);

    nbt_write_u8(&nbt, NBT_END); /* end root */

    free(block_data);

    /* Compress with zlib (raw deflate, then wrap in gzip manually) */
    mz_ulong compressed_size = mz_compressBound((mz_ulong)nbt.size);
    uint8_t* compressed = (uint8_t*)malloc(compressed_size + 64);

    /* Write gzip header */
    size_t gzip_pos = 0;
    compressed[gzip_pos++] = 0x1F; /* magic */
    compressed[gzip_pos++] = 0x8B;
    compressed[gzip_pos++] = 0x08; /* deflate */
    compressed[gzip_pos++] = 0x00; /* flags */
    compressed[gzip_pos++] = 0; compressed[gzip_pos++] = 0;
    compressed[gzip_pos++] = 0; compressed[gzip_pos++] = 0; /* mtime */
    compressed[gzip_pos++] = 0x00; /* xfl */
    compressed[gzip_pos++] = 0xFF; /* OS unknown */

    /* Deflate the NBT data (raw, no zlib header) */
    mz_stream stream;
    memset(&stream, 0, sizeof(stream));
    stream.next_in = nbt.data;
    stream.avail_in = (unsigned int)nbt.size;
    stream.next_out = compressed + gzip_pos;
    stream.avail_out = (unsigned int)(compressed_size);

    if (mz_deflateInit2(&stream, MZ_DEFAULT_COMPRESSION, MZ_DEFLATED,
                         -15, /* negative = raw deflate, no header */
                         8, MZ_DEFAULT_STRATEGY) != MZ_OK) {
        fprintf(stderr, "[Schem] Failed to init deflate\n");
        free(nbt.data);
        free(compressed);
        return -1;
    }

    mz_deflate(&stream, MZ_FINISH);
    mz_deflateEnd(&stream);
    gzip_pos += stream.total_out;

    /* Write gzip footer: CRC32 + original size */
    mz_ulong crc = mz_crc32(MZ_CRC32_INIT, nbt.data, (size_t)nbt.size);
    compressed[gzip_pos++] = (uint8_t)(crc & 0xFF);
    compressed[gzip_pos++] = (uint8_t)((crc >> 8) & 0xFF);
    compressed[gzip_pos++] = (uint8_t)((crc >> 16) & 0xFF);
    compressed[gzip_pos++] = (uint8_t)((crc >> 24) & 0xFF);
    uint32_t orig_size = (uint32_t)nbt.size;
    compressed[gzip_pos++] = (uint8_t)(orig_size & 0xFF);
    compressed[gzip_pos++] = (uint8_t)((orig_size >> 8) & 0xFF);
    compressed[gzip_pos++] = (uint8_t)((orig_size >> 16) & 0xFF);
    compressed[gzip_pos++] = (uint8_t)((orig_size >> 24) & 0xFF);

    size_t final_size = gzip_pos;

    free(nbt.data);

    /* Write to file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Schem] Cannot open %s for writing\n", path);
        free(compressed);
        return -1;
    }

    fwrite(compressed, 1, final_size, f);
    fclose(f);
    free(compressed);

    printf("[Schem] Exported %dx%dx%d (%d palette, %zu bytes) to %s\n",
           width, height, length, palette_count, final_size, path);

    return palette_count;
}
